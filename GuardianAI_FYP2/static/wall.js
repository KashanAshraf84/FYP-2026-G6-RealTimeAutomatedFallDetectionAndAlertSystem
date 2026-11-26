"use strict";
/*
 * Monitor wall live previews.
 *  - Webcam cameras running in this tab (started on their camera pages) are shown live in
 *    their cards, all at once, with each person's box; frames keep going to the model.
 *  - Any other live source (another tab, another device) is shown from the server's latest
 *    in-memory frame (/cameras/<id>/snapshot/) with the boxes the model last reported.
 */
(()=>{
 const grid=document.querySelector('.camera-grid');if(!grid||!window.GuardianCapture)return;
 const G=window.GuardianCapture,liveBtn=document.getElementById('wall-live');
 const cards=new Map([...grid.querySelectorAll('[data-camera]')].map(c=>[Number(c.dataset.camera),c]));
 const views=new Map(); // camera id -> {layer, overlay, kind:'local'|'remote', stop()}
 const stale=new Set(); // cameras stopped here whose server telemetry has not gone offline yet

 function layerFor(card,media){
  const preview=card.querySelector('.camera-preview'),layer=document.createElement('div'),canvas=document.createElement('canvas');
  layer.className='camera-live';layer.append(media,canvas);preview.prepend(layer);preview.classList.add('is-live');
  return {layer,canvas};
 }
 function drop(id){const v=views.get(id);if(!v)return;v.stop?.();v.layer.remove();cards.get(id)?.querySelector('.camera-preview').classList.remove('is-live');views.delete(id);}
 function setCard(id,result){
  const card=cards.get(id);if(!card)return;const labels=new Set((result.predictions||[]).map(p=>p.label));
  const status=result.ready===false?'preview':['fall','warning','normal','unknown'].find(l=>labels.has(l))||'no_subject';
  card.querySelector('.camera-status').textContent=G.statusText(status,result);
  card.querySelector('.camera-subjects').textContent=(result.predictions?.length||0)+' subjects';
 }
 function liveButton(){
  if(!liveBtn)return;const ids=[...views].filter(([,v])=>v.kind==='local').map(([id])=>id);
  liveBtn.hidden=!ids.length;
  if(ids.length)liveBtn.querySelector('span').textContent=ids.length===1?cards.get(ids[0]).querySelector('.camera-caption h3').textContent+' · Stop':ids.length+' cameras · Stop all';
 }

 // ---- webcam cameras of this tab
 async function showLocal(id,device){
  const card=cards.get(id);
  const video=document.createElement('video'),{layer,canvas}=layerFor(card,video);
  views.set(id,{layer,kind:'local',stop:()=>G.stop(id,{forget:false})});
  try{await G.start(id,{video,canvas,device,onResult:r=>setCard(id,r),onError:msg=>{toast(msg);drop(id);stale.add(id);liveButton();}});}
  catch(e){drop(id);G.live.remove(id);toast('Webcam could not resume: '+e.message);}
  liveButton();
 }
 liveBtn?.addEventListener('click',()=>{for(const [id,v] of [...views])if(v.kind==='local'){G.stop(id);drop(id);stale.add(id);}liveButton();});
 for(const [key,device] of Object.entries(G.live.all())){
  const id=Number(key),card=cards.get(id);
  if(card&&card.dataset.canCapture==='1'&&card.dataset.kind==='browser'){G.claim(id);showLocal(id,device);}
 }

 // ---- sources live elsewhere: server snapshots
 function remoteView(id,card){
  const img=document.createElement('img');img.alt='';const {layer,canvas}=layerFor(card,img),overlay=new G.Overlay(canvas,img);
  const v={layer,overlay,kind:'remote',timer:0,alive:true,misses:0};
  const next=()=>{if(!v.alive)return;img.src=card.dataset.snapshotUrl+'?t='+Date.now();};
  img.onload=()=>{v.misses=0;v.timer=setTimeout(next,250);};
  // No fresh frame (source stopped): drop the picture instead of showing a frozen one.
  img.onerror=()=>{if(++v.misses>=2){stale.add(id);drop(id);return;}v.timer=setTimeout(next,700);};
  v.stop=()=>{v.alive=false;clearTimeout(v.timer);overlay.stop();};
  overlay.start();next();views.set(id,v);return v;
 }
 async function poll(){
  if(document.hidden)return;
  try{
   const r=await fetch('/api/status/',{headers:{Accept:'application/json'}});if(!r.ok||r.redirected)return;const data=await r.json();
   for(const c of data.cameras){
    const card=cards.get(c.id);if(!card||views.get(c.id)?.kind==='local')continue;
    let live=c.status!=='offline'&&c.status!=='preview'&&c.kind!=='worker';
    if(!live)stale.delete(c.id);else if(stale.has(c.id))live=false;
    let v=views.get(c.id);
    if(live&&!v)v=remoteView(c.id,card);
    if(!live&&v){drop(c.id);continue;}
    if(v)v.overlay.update(c.result.predictions,1000);
   }
  }catch{/* app.js shows connection problems */}
 }
 poll();const pollTimer=setInterval(poll,1000);
 window.addEventListener('pagehide',()=>{clearInterval(pollTimer);for(const v of views.values())if(v.kind==='remote')v.stop();});
})();
