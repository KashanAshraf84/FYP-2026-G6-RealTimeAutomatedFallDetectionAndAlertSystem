"""
Fall Detection System - Main Entry Point
==========================================
Provides CLI interface for training, inference, and system management.
"""

import argparse
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig, CONFIG


def cmd_train(args):
    """Run training pipeline."""
    from train import train_pipeline

    config = SystemConfig()
    config.model.architecture = args.arch
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.lr

    train_pipeline(
        data_path=args.data,
        use_synthetic=args.synthetic or (args.data is None),
        config=config,
    )


def cmd_detect(args):
    """Run real-time fall detection."""
    from inference import FallDetectionInference

    config = SystemConfig()

    # Determine video source
    source = args.source
    if source is not None:
        try:
            source = int(source)  # Camera index
        except ValueError:
            pass  # File path

    detector = FallDetectionInference(
        config=config,
        model_path=args.model,
    )

    detector.run_on_video(source=source)


def cmd_preprocess(args):
    """Preprocess video dataset."""
    from dataset import preprocess_dataset_folder
    from pose_estimator import PoseEstimator

    config = SystemConfig()
    pose_est = PoseEstimator(config.pose)

    preprocess_dataset_folder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pose_estimator=pose_est,
        feature_config=config.features,
    )

    pose_est.release()
    print("Preprocessing complete!")


def cmd_export(args):
    """Export model to ONNX."""
    from train import FallDetectionTrainer

    config = SystemConfig()
    config.model.architecture = args.arch
    trainer = FallDetectionTrainer(config)
    trainer.load_checkpoint(args.model)
    trainer.export_onnx(args.output)
    print(f"Model exported to {args.output}")


def cmd_demo(args):
    """
    Run a demonstration with synthetic data:
    trains a model, then runs inference on webcam.
    """
    from train import train_pipeline
    from inference import FallDetectionInference

    config = SystemConfig()
    config.model.architecture = args.arch
    config.model.epochs = 30  # Quick training for demo

    print("=" * 60)
    print("  FALL DETECTION SYSTEM - DEMO MODE")
    print("=" * 60)

    # Step 1: Quick training with synthetic data
    print("\n📋 Step 1: Training model with synthetic data...")
    trainer, results = train_pipeline(use_synthetic=True, config=config)

    # Step 2: Run inference
    print("\n📹 Step 2: Starting real-time detection...")
    detector = FallDetectionInference(config=config)

    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    detector.run_on_video(source=source)


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection System - Pose Estimation & Video Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with synthetic data (for demo/testing)
  python main.py train --synthetic

  # Train with real video data
  python main.py train --data ./data/videos

  # Train with preprocessed data
  python main.py train --data ./data/processed/dataset.npz

  # Run real-time detection (webcam)
  python main.py detect

  # Run detection on a video file
  python main.py detect --source path/to/video.mp4

  # Preprocess a video dataset
  python main.py preprocess --data-dir ./raw_videos --output-dir ./data/processed

  # Export trained model to ONNX
  python main.py export --model models/fall_detector_best.pth --output models/fall_detector.onnx

  # Run full demo (train + detect)
  python main.py demo
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train the fall detection model")
    train_parser.add_argument("--data", type=str, default=None, help="Path to dataset (.npz or video folder)")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    train_parser.add_argument("--arch", type=str, default="cnn_lstm", choices=["lstm", "cnn_lstm", "transformer"])
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.set_defaults(func=cmd_train)

    # --- Detect ---
    detect_parser = subparsers.add_parser("detect", help="Run real-time detection")
    detect_parser.add_argument("--source", type=str, default=None, help="Video source (camera index or file path)")
    detect_parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    detect_parser.set_defaults(func=cmd_detect)

    # --- Preprocess ---
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess video dataset")
    preprocess_parser.add_argument("--data-dir", type=str, required=True, help="Input video directory")
    preprocess_parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # --- Export ---
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    export_parser.add_argument("--output", type=str, default="models/fall_detector.onnx")
    export_parser.add_argument("--arch", type=str, default="cnn_lstm", choices=["lstm", "cnn_lstm", "transformer"])
    export_parser.set_defaults(func=cmd_export)

    # --- Demo ---
    demo_parser = subparsers.add_parser("demo", help="Run full demo (train + detect)")
    demo_parser.add_argument("--source", type=str, default=None, help="Video source")
    demo_parser.add_argument("--arch", type=str, default="cnn_lstm", choices=["lstm", "cnn_lstm", "transformer"])
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
