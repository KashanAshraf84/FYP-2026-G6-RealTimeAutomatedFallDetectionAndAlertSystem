"use strict";
/*
 * Camera page: browser webcam / local video -> sampled JPEG frames -> Django -> detector -> overlay
 * (capture code in capture.js). A webcam keeps running when you open other pages and resumes here
 * when you come back; only Stop ends it. Several webcam cameras can be live at once.
 */
(()=>{
 const monitor=document.getElementById('monitor');if(!monitor)return;
 const G=window.GuardianCapture,cameraId=Number(monitor.dataset.cameraId);
 const video=document.getElementById('source-video'), canvas=document.getElementById('overlay'), state=document.getElementById('source-state'), message=document.getElementById('inference-message'), subjects=document.getElementById('subjects'), frameInfo=document.getElementById('frame-info'), stopButton=document.getElementById('stop-source'), startWebcam=document.getElementById('start-webcam'), startVideo=document.getElementById('start-video'), fileInput=document.getElementById('video-file'), placeholder=document.getElementById('video-placeholder'), deviceSelect=document.getElementById('webcam-device');
 let objectUrl=null,fileSender=null,webcamOn=false;
 function subjectCard(p){
  let card=subjects.querySelector(`[data-track="${CSS.escape(p.track_id)}"]`);
  if(!card){card=document.createElement('div');card.className='subject-card';card.dataset.track=p.track_id;card.append(document.createElement('h3'),document.createElement('p'),document.createElement('pre'));subjects.append(card);}
  card.querySelector('h3').textContent=p.track_id+' · '+G.labelText(p);
  card.querySelector('p').textContent=G.isPartial(p)?'Move back so hips and legs are in view. The model needs the whole body.':p.label==='unknown'?'Collecting 1.5 s of motion history':'Model score: '+(p.confidence*100).toFixed(1)+'%';
  const ev=JSON.stringify(p.evidence,null,2);if(card.querySelector('pre').textContent!==ev)card.querySelector('pre').textContent=ev;
 }
 function render(result,fps,latency){
  message.textContent=result.ready?(result.message||'Model connected · '+result.model_version):result.message||'Model is not ready.';
  state.textContent=result.ready?'ANALYSING':'PREVIEW ONLY';
  frameInfo.textContent=latency+' ms processing · '+result.predictions.length+' subjects · '+(fps?fps.toFixed(1):'—')+' frames/s analysed';
  if(result.predictions.length){subjects.className='';const keep=new Set();for(const p of result.predictions){subjectCard(p);keep.add(p.track_id);}for(const c of [...subjects.children])if(!keep.has(c.dataset.track))c.remove();}
  else{subjects.className='empty';subjects.replaceChildren();subjects.textContent=result.ready?'No subjects returned by the model.':'Waiting for a trained detector.';}
 }
 function idle(){
  state.textContent='STOPPED';if(stopButton)stopButton.disabled=true;if(startWebcam)startWebcam.disabled=false;if(startVideo)startVideo.disabled=!objectUrl;placeholder.hidden=false;webcamOn=false;
 }
 function busy(){placeholder.hidden=true;stopButton.disabled=false;if(startWebcam)startWebcam.disabled=true;if(startVideo)startVideo.disabled=true;state.textContent='CONNECTING';message.textContent='Connecting to the model…';}

 // ---- webcam (shared live set)
 async function listDevices(){
  if(!deviceSelect||!navigator.mediaDevices?.enumerateDevices)return;
  const cams=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
  const chosen=deviceSelect.value||G.live.all()[cameraId]||'';
  deviceSelect.replaceChildren(...cams.map((d,i)=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||'Camera '+(i+1);return o;}));
  if(chosen&&cams.some(d=>d.deviceId===chosen))deviceSelect.value=chosen;
  deviceSelect.hidden=cams.length<2;
 }
 async function startCamera(resumed){
  busy();
  try{
   await G.start(cameraId,{video,canvas,device:(deviceSelect&&!deviceSelect.hidden&&deviceSelect.value)||G.live.all()[cameraId]||'',
    onResult:(r,fps)=>render(r,fps,r.latency_ms),onError:msg=>{message.textContent=msg;idle();}});
   webcamOn=true;if(resumed)message.textContent='Webcam resumed. It keeps running on other pages until you press Stop.';
   listDevices();
  }catch(e){message.textContent='Camera could not start: '+e.message;G.live.remove(cameraId);idle();}
 }
 startWebcam?.addEventListener('click',()=>startCamera(false));
 deviceSelect?.addEventListener('change',()=>{if(webcamOn)startCamera(false);});
 if(startWebcam){listDevices();if(G.live.has(cameraId)){G.claim(cameraId);startCamera(true);}}

 // ---- local video file (this page only)
 function stopFile(){fileSender?.stop();fileSender=null;video.pause();}
 fileInput?.addEventListener('change',()=>{stopFile();idle();if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=null;const file=fileInput.files[0];if(file){objectUrl=URL.createObjectURL(file);video.src=objectUrl;video.load();startVideo.disabled=false;message.textContent='Selected '+file.name+'. Start analysis to send sampled frames to the server.';}});
 startVideo?.addEventListener('click',async()=>{
  try{video.currentTime=0;await video.play();busy();
   fileSender=G.createSender({video,overlay:new G.Overlay(canvas,video),frameUrl:monitor.dataset.frameUrl,csrf:monitor.querySelector('[name=csrfmiddlewaretoken]').value,
    onResult:(r,fps)=>{render(r,fps,r.latency_ms);if(r.event_ids?.length)toast('New incident recorded: #'+r.event_ids.join(', #'));},
    onError:msg=>{message.textContent=msg;stopFile();idle();},onStop:()=>{stopFile();idle();}});
   fileSender.start();
  }catch(e){message.textContent='Video could not play: '+e.message;stopFile();idle();}
 });
 video.addEventListener('ended',()=>{if(fileSender){stopFile();idle();message.textContent='Video finished. Recorded incidents are available in Event history.';}});

 stopButton?.addEventListener('click',()=>{if(webcamOn)G.stop(cameraId);stopFile();idle();message.textContent='Source stopped.';});
 if(monitor.dataset.kind==='worker'){
  const pollWorker=async()=>{try{const response=await fetch('/api/status/');if(!response.ok||response.redirected)throw Error('Connection lost');const data=await response.json();const camera=data.cameras.find(c=>c.id===cameraId);if(camera&&camera.status!=='offline'&&camera.result.model_version){render(camera.result,0,'—');placeholder.querySelector('h2').textContent='Worker connected';placeholder.querySelector('p').textContent='Telemetry is live. Video remains on the server.';}else{state.textContent='OFFLINE';message.textContent='No recent frames from this worker. Start the Python camera worker on the server.';subjects.textContent='Waiting for worker telemetry.';}}catch{state.textContent='DISCONNECTED';message.textContent='Cannot reach server. Values may be stale.';}};
  pollWorker();const workerTimer=setInterval(pollWorker,3000);window.addEventListener('pagehide',()=>clearInterval(workerTimer));
 }
 window.addEventListener('pagehide',()=>{stopFile();if(objectUrl)URL.revokeObjectURL(objectUrl);});
})();
