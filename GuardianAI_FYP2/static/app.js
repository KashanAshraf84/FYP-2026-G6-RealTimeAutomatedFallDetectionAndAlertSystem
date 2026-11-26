"use strict";
const toast = message => { const el=document.getElementById('toast'); if(el){el.textContent=message;el.hidden=false;setTimeout(()=>el.hidden=true,5000);} };
try { if(localStorage.getItem('guardian-theme')==='dark') document.body.classList.add('dark'); } catch {}
document.getElementById('theme-toggle')?.addEventListener('click',()=>{document.body.classList.toggle('dark');try{localStorage.setItem('guardian-theme',document.body.classList.contains('dark')?'dark':'light');}catch{}});
const clock=document.getElementById('clock');if(clock){const tick=()=>clock.textContent=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});tick();setInterval(tick,1000);}
document.getElementById('enable-notifications')?.addEventListener('click',async()=>{if(!('Notification'in window)){toast('Browser notifications are not supported.');return;}const p=await Notification.requestPermission();try{localStorage.setItem('guardian-notifications',p==='granted'?'yes':'no');}catch{}toast(p==='granted'?'Alerts enabled while the workspace is open.':'Browser permission was not granted.');});
let previousIds=null;
async function refresh(){
 try{
  const response=await fetch('/api/status/',{headers:{Accept:'application/json'}});if(!response.ok||response.redirected)throw Error('Session ended');const data=await response.json();
  const set=(id,value)=>{const e=document.getElementById(id);if(e)e.textContent=value;};
  set('kpi-cameras',data.cameras.filter(c=>c.status!=='offline').length+' / '+data.cameras.length);
  set('kpi-subjects',data.cameras.reduce((n,c)=>n+(c.result.predictions?.length||0),0));set('kpi-events',data.open_events);set('kpi-falls',data.falls_today);
  for(const c of data.cameras){const card=document.querySelector(`[data-camera="${c.id}"]`);if(!card)continue;card.querySelector('.camera-status').textContent=window.GuardianCapture?window.GuardianCapture.statusText(c.status,c.result):c.status.replace('_',' ').toUpperCase();card.querySelector('.camera-subjects').textContent=(c.result.predictions?.length||0)+' subjects';card.querySelector('.camera-message').textContent=c.status==='offline'?'Open source to begin monitoring':c.result.message||'Source telemetry connected';}
  const list=document.getElementById('alert-list');if(list&&data.alerts.length){list.replaceChildren();for(const e of data.alerts){const a=document.createElement('a');a.className='alert-item';a.href='/events/'+e.id+'/';const badge=document.createElement('span');badge.className='pill danger';badge.textContent=(e.demo?'DEMO · ':'')+'FALL';const h=document.createElement('h3');h.textContent=e.camera;const p=document.createElement('p');p.textContent=e.time+' · score '+e.confidence.toFixed(2);const action=document.createElement('span');action.textContent='Review incident →';a.append(badge,h,p,action);list.append(a);}}
  if(list&&!data.alerts.length){list.replaceChildren();const d=document.createElement('div');d.className='empty';const h=document.createElement('h3');h.textContent='Nothing awaiting review';const p=document.createElement('p');p.textContent='New incidents will appear here when an alert is recorded.';d.append(h,p);list.append(d);}
  if(previousIds){for(const e of data.alerts){let enabled=false;try{enabled=localStorage.getItem('guardian-notifications')==='yes';}catch{}if(!e.demo&&!previousIds.has(e.id)&&enabled&&'Notification'in window&&Notification.permission==='granted'){const n=new Notification('GuardianAI: possible fall',{body:e.camera+' · review incident #'+e.id,tag:'guardian-'+e.id});n.onclick=()=>{window.focus();location.href='/events/'+e.id+'/';};}}}
  previousIds=new Set(data.alerts.map(e=>e.id));set('sync-label','Updated '+new Date().toLocaleTimeString());
 }catch{const el=document.getElementById('sync-label');if(el)el.textContent='Connection lost · displayed values may be stale';}
}
if(document.querySelector('.app-shell')){refresh();setInterval(refresh,3000);}
