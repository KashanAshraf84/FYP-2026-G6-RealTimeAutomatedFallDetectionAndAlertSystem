"use strict";
/*
 * Live capture shared by every page (loaded from base.html).
 *
 * Overlay  draws each person's box + "P1 · NORMAL 98%" tag over a <video> or <img>
 *          (object-fit: contain) and animates boxes between model replies.
 * Sender   sends sampled JPEG frames of a <video> to a camera's frame URL. The server
 *          answers with next_ms, the wait before the next frame, so all live cameras
 *          together never keep the model busy more than half the time.
 * Live set Webcam cameras started in this tab are remembered in sessionStorage
 *          ({cameraId: deviceId}). Every page resumes them: the camera page and the
 *          monitor wall show them, other pages keep them running in the background,
 *          so monitoring continues while you review incidents. Only Stop ends one.
 */
window.GuardianCapture=(()=>{
 const TARGET_MS=160, MAX_IN_FLIGHT=2, LIVE_KEY='guardian-live-cams';
 const COLORS={fall:'#ff6b91',warning:'#f9c65a',normal:'#43ddb0',unknown:'#a0a8b8'};
 const isPartial=p=>p.label==='unknown'&&p.evidence?.status==='partial_view';
 const labelText=p=>p.label!=='unknown'?p.label.toUpperCase():isPartial(p)?'PARTIAL VIEW':'WARMING UP';
 const tagText=p=>p.track_id+' · '+labelText(p)+(p.label==='unknown'?'':' '+Math.round(p.confidence*100)+'%');
 /** Card status for a camera's latest result ("FALL", "PARTIAL VIEW", ...). */
 function statusText(status,result){
  if(status!=='unknown')return String(status).replace('_',' ').toUpperCase();
  const ps=result?.predictions||[];return ps.length&&ps.every(isPartial)?'PARTIAL VIEW':'WARMING UP';
 }
 const boxAt=(tw,now)=>{const k=Math.min(1,Math.max(0,(now-tw.t0)/(tw.t1-tw.t0||1)));return tw.from.map((v,i)=>v+(tw.to[i]-v)*k);};

 class Overlay{
  constructor(canvas,media){this.canvas=canvas;this.media=media;this.tweens=new Map();this.raf=0;}
  update(predictions,durationMs){
   const now=performance.now(),live=new Set();
   for(const p of predictions||[]){if(!p.bbox)continue;live.add(p.track_id);const prev=this.tweens.get(p.track_id);this.tweens.set(p.track_id,{from:prev?boxAt(prev,now):p.bbox,to:p.bbox,t0:now,t1:now+durationMs,label:p.label,tag:tagText(p)});}
   for(const id of [...this.tweens.keys()])if(!live.has(id))this.tweens.delete(id);
  }
  start(){if(!this.raf){const tick=()=>{this.draw();this.raf=requestAnimationFrame(tick);};this.raf=requestAnimationFrame(tick);}}
  stop(){cancelAnimationFrame(this.raf);this.raf=0;this.tweens.clear();this.clear();}
  clear(){const c=this.canvas;c.getContext('2d').clearRect(0,0,c.width,c.height);}
  draw(){
   const m=this.media,mw=m.videoWidth||m.naturalWidth,mh=m.videoHeight||m.naturalHeight;if(!mw||!mh)return;
   const box=this.canvas.getBoundingClientRect();if(!box.width)return;
   if(this.canvas.width!==Math.round(box.width)||this.canvas.height!==Math.round(box.height)){this.canvas.width=Math.round(box.width);this.canvas.height=Math.round(box.height);}
   const ctx=this.canvas.getContext('2d');ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
   if(!this.tweens.size)return;
   const s=Math.min(box.width/mw,box.height/mh),w=mw*s,h=mh*s,x=(box.width-w)/2,y=(box.height-h)/2,now=performance.now(),small=box.width<480,fs=small?11:13,th=small?17:20;
   for(const tw of this.tweens.values()){
    const [x1,y1,x2,y2]=boxAt(tw,now),bx=x+x1*w,by=y+y1*h,color=COLORS[tw.label]||COLORS.unknown;
    ctx.strokeStyle=color;ctx.lineWidth=tw.label==='fall'?3:2;ctx.strokeRect(bx,by,(x2-x1)*w,(y2-y1)*h);
    ctx.font=`bold ${fs}px sans-serif`;const tw2=ctx.measureText(tw.tag).width+10,ty=by>=th?by-th:by;
    ctx.fillStyle=color;ctx.fillRect(bx,ty,tw2,th);ctx.fillStyle='#111318';ctx.fillText(tw.tag,bx+5,ty+th-6);
   }
  }
 }

 /** Sender: o = {video, overlay (Overlay), frameUrl, csrf, onResult(result, fps), onError(message), onStop()} */
 function createSender(o){
  const capture=document.createElement('canvas'),controllers=new Set();
  let running=false,session=0,timer=0,inFlight=0,interval=TARGET_MS,seq=0,lastSeq=0,arrivals=[];
  async function send(my){
   if(!running||my!==session)return;
   const v=o.video;if(v.paused||v.ended){api.stop();o.onStop?.();return;}
   timer=setTimeout(()=>send(my),interval);
   if(!v.videoWidth||inFlight>=MAX_IN_FLIGHT)return;
   const sc=Math.min(640/v.videoWidth,1080/v.videoHeight);capture.width=Math.max(1,Math.round(v.videoWidth*sc));capture.height=Math.max(1,Math.round(v.videoHeight*sc));capture.getContext('2d').drawImage(v,0,0,capture.width,capture.height);
   const blob=await new Promise(r=>capture.toBlob(r,'image/jpeg',.8));if(!running||my!==session||!blob)return;
   const controller=new AbortController();controllers.add(controller);const timeout=setTimeout(()=>controller.abort(),60000);const mySeq=++seq;inFlight++;
   try{
    const response=await fetch(o.frameUrl,{method:'POST',headers:{'Content-Type':'image/jpeg','X-CSRFToken':o.csrf},body:blob,signal:controller.signal});
    if(response.redirected)throw Error('Your session ended. Sign in again.');
    if(response.status===403)throw Error('Your account cannot start camera capture.');
    const result=await response.json();if(!response.ok)throw Error(result.error||'Frame processing failed');
    if(my!==session||mySeq<lastSeq)return;lastSeq=mySeq;
    interval=Math.min(1000,Math.max(TARGET_MS,Number(result.next_ms)||TARGET_MS));
    const now=performance.now();arrivals.push(now);arrivals=arrivals.filter(t=>now-t<3000);
    o.overlay?.update(result.predictions,interval);
    o.onResult?.(result,arrivals.length>1?(arrivals.length-1)/((now-arrivals[0])/1000):0);
   }catch(error){if(my!==session||(error.name==='AbortError'&&!running))return;api.stop();o.onError?.(error.name==='AbortError'?'Inference timed out. Check the server and restart.':error.message);}
   finally{clearTimeout(timeout);controllers.delete(controller);inFlight=Math.max(0,inFlight-1);}
  }
  const api={
   get running(){return running;},
   start(){running=true;session++;inFlight=0;seq=lastSeq=0;arrivals=[];interval=TARGET_MS;o.overlay?.start();send(session);},
   stop(){running=false;session++;clearTimeout(timer);for(const c of controllers)c.abort();controllers.clear();inFlight=0;o.overlay?.stop();},
  };
  return api;
 }

 // ---- live webcam cameras of this tab
 const store={
  all(){try{return JSON.parse(sessionStorage.getItem(LIVE_KEY))||{};}catch{return {};}},
  save(map){try{sessionStorage.setItem(LIVE_KEY,JSON.stringify(map));}catch{}},
  has(id){return String(id) in this.all();},
  set(id,device){const m=this.all();m[id]=device||'';this.save(m);},
  remove(id){const m=this.all();delete m[id];this.save(m);},
  clear(){try{sessionStorage.removeItem(LIVE_KEY);}catch{}},
 };
 const running=new Map(), claimed=new Set(); // camera id -> {stream, video, sender, overlay}
 const csrf=()=>document.querySelector('[name=csrfmiddlewaretoken]')?.value||'';
 function openWebcam(device){
  if(!navigator.mediaDevices?.getUserMedia)return Promise.reject(Error('Webcam access requires localhost or HTTPS.'));
  const video={width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}};if(device)video.deviceId={exact:device};
  return navigator.mediaDevices.getUserMedia({video,audio:false}).catch(e=>{if(device&&e.name==='OverconstrainedError')return openWebcam('');throw e;});
 }
 /** Start (or resume) a webcam camera. o = {video, canvas, device, onResult(result, fps), onError(message)}; video/canvas optional (background). */
 async function start(id,o={}){
  id=Number(id);if(running.has(id))stop(id,{forget:false});
  const stream=await openWebcam(o.device);
  const video=o.video||document.createElement('video');video.muted=true;video.playsInline=true;video.srcObject=stream;
  try{await video.play();}catch(e){stream.getTracks().forEach(t=>t.stop());throw e;}
  const overlay=o.canvas?new Overlay(o.canvas,video):null;
  const sender=createSender({video,overlay,frameUrl:`/cameras/${id}/frame/`,csrf:csrf(),
   onResult:(r,fps)=>{if(r.event_ids?.length&&typeof toast==='function')toast('New incident recorded: #'+r.event_ids.join(', #'));o.onResult?.(r,fps);},
   onError:msg=>{stop(id);o.onError?.(msg);}});
  running.set(id,{stream,video,sender,overlay});store.set(id,o.device);sender.start();pill();
  return running.get(id);
 }
 /** Stop a camera in this page. forget:false keeps it in the live set (page change), so the next page resumes it. */
 function stop(id,{forget=true}={}){
  id=Number(id);const r=running.get(id);
  if(r){r.sender.stop();r.stream.getTracks().forEach(t=>t.stop());r.video.srcObject=null;running.delete(id);}
  if(forget)store.remove(id);pill();
 }
 function stopAll(){for(const id of Object.keys(store.all()))stop(id);}
 function pill(){
  const el=document.getElementById('live-pill');if(!el)return;const n=Object.keys(store.all()).length;
  el.hidden=!n;el.textContent=n?`● LIVE · ${n} camera${n>1?'s':''}`:'';
 }
 // Pages that display a camera claim it while their script runs; everything else resumes in the background.
 document.addEventListener('DOMContentLoaded',()=>{
  for(const [key,device] of Object.entries(store.all())){
   const id=Number(key);if(claimed.has(id)||running.has(id))continue;
   start(id,{device,onError:()=>{}}).catch(()=>store.remove(id));
  }
  pill();
  document.querySelectorAll('form[action$="/logout/"]').forEach(f=>f.addEventListener('submit',()=>store.clear()));
 });
 window.addEventListener('pagehide',()=>{for(const id of [...running.keys()])stop(id,{forget:false});});
 return {Overlay,createSender,start,stop,stopAll,claim:id=>claimed.add(Number(id)),running,live:store,statusText,labelText,isPartial,COLORS};
})();
