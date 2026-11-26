"use strict";
// Adds a show/hide button to every password field on the page.
(()=>{
 const eye='<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
 const eyeOff='<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3l18 18"/><path d="M10.6 5.1A10.4 10.4 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.8 8.4 2 12 2 12s3.6 7 10 7a9.9 9.9 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>';
 for(const input of document.querySelectorAll('input[type=password]')){
  const wrap=document.createElement('div');wrap.className='password-wrap';input.parentNode.insertBefore(wrap,input);wrap.append(input);
  const btn=document.createElement('button');btn.type='button';btn.className='password-toggle';btn.innerHTML=eye;btn.setAttribute('aria-label','Show password');btn.title='Show password';
  btn.addEventListener('click',()=>{const show=input.type==='password';input.type=show?'text':'password';btn.innerHTML=show?eyeOff:eye;const label=show?'Hide password':'Show password';btn.setAttribute('aria-label',label);btn.title=label;input.focus();});
  wrap.append(btn);
 }
})();
