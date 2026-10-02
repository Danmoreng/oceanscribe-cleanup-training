(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const state = { csrf: '', sessionId:'', status: {}, statusReady: false, card: null, count: Number(localStorage.getItem('os_review_session_count')||0), paused: localStorage.getItem('os_review_paused')==='true', recordingTasks: [], task: null, recorder: null, chunks: [], stream: null, audioSettings: {}, recordingFailed: false, startedAt: 0, timer: null, recordingId: null, audioContext: null, analyser: null, levelFrame: null, decisionBusy: false, conflicted: false, pendingDecision: false, polling: false, pollTimer: null };
  let statusPromise=null;
  const esc = (s) => String(s ?? '');
  const id = () => crypto.randomUUID();
  const text = (node, value) => { node.textContent = esc(value); };
  function studioNotice(message,kind='neutral'){const node=$('#studioStatus');if(!node)return;text(node,message);node.className=`studio-status ${kind}`;}
  async function api(path, body) {
    if(body!==undefined&&(!state.csrf||!state.sessionId))await loadStatus();
    const options = { method: body === undefined ? 'GET' : 'POST', headers: {} };
    if (body !== undefined) { options.headers['Content-Type'] = 'application/json'; options.headers['X-CSRF-Token'] = state.csrf; options.headers['X-Local-Session'] = state.sessionId; options.body = JSON.stringify(body); }
    let response;
    try{response=await fetch(path, options);}catch(error){studioNotice(`Verbindungsfehler: ${error.message}`,'error');throw error;}
    const result = await response.json();
    if (!response.ok) { const error=new Error(result.message || `HTTP ${response.status}`); error.status=response.status; error.code=result.error; studioNotice(error.message,'error'); throw error; }
    return result;
  }
  function flash(message, bad = false) {
    const node = $('#saveState');
    if (node) { text(node, message); node.classList.toggle('error-text', bad); }
    if (bad) { text($('#recordStatus'), message); studioNotice(message,'error'); }
  }
  function getStatus(payload) {
    state.csrf = payload.csrf_token || state.csrf;
    state.sessionId=payload.session_id||state.sessionId;
    state.statusReady=Boolean(state.csrf&&state.sessionId);
    state.status = payload.status || payload;
    if (payload.asr) {
      const runner = payload.asr;
      const engines = runner.engines || {};
      const message = Object.values(engines).map((engine) => `${engine.actual_model || engine.engine || 'ASR'}: ${engine.available ? 'bereit' : (engine.error || 'nicht verfügbar')}`).join(' · ') || runner.target_mismatch || runner.error || 'Lokale ASR-Integration wird geprüft.';
      text($('#asrStatus'), message);
      $('#asrStatus').classList.toggle('warning-notice', !Object.values(engines).some((engine) => engine.available));
      if (engines.nemotron) $('#asrEngine option[value="nemotron"]').disabled = !engines.nemotron.available;
      if (engines.parakeet) $('#asrEngine option[value="parakeet"]').disabled = !engines.parakeet.available;
      if (engines.nemotron || engines.parakeet) $('#asrEngine option[value="both"]').disabled = !engines.nemotron?.available && !engines.parakeet?.available;
    }
    if(state.statusReady){studioNotice('Lokal verbunden · Kampagnenstatus geladen.','ready');$('#refreshReview').disabled=false;if($('#refreshReview').textContent.includes('verbunden'))text($('#refreshReview'),state.paused?'Fünferblock fortsetzen':'Fünferblock beginnen / fortsetzen');}
    renderBudgets(); renderStats();
  }
  async function loadStatus(force=false){
    if(state.csrf&&state.sessionId&&!force)return state.status;
    if(statusPromise)return statusPromise;
    studioNotice('Verbinde mit lokalem Studio und lade Kampagnenstatus …','loading');text($('#refreshReview'),'Studio wird verbunden …');$('#refreshReview').disabled=true;
    statusPromise=(async()=>{try{const payload=await api('/api/status');getStatus(payload);if(!state.statusReady)throw new Error('Die lokale Sitzung konnte nicht eingerichtet werden (Session- oder CSRF-Token fehlt).');return payload.status||payload;}catch(error){state.statusReady=false;studioNotice(`Studio konnte nicht geladen werden: ${error.message}. Du kannst die Verbindung erneut versuchen.`,'error');text($('#refreshReview'),'Erneut verbinden & Fünferblock starten');$('#refreshReview').disabled=false;throw error;}finally{statusPromise=null;}})();
    return statusPromise;
  }
  function budget(kind) { return state.status?.budgets?.[kind] || { used: 0, limit: kind === 'reviews' ? 50 : 10, maximum: kind === 'reviews' ? 100 : 20 }; }
  function renderBudgets() {
    const r = budget('reviews'), a = budget('recordings');
    text($('#reviewBudget'), `${r.used} von ${r.limit} Reviews verbraucht`);
    text($('#recordBudget'), `${a.used} von ${a.limit} Versuchen verbraucht`);
    text($('#cardRemaining'), `${Math.max(0, r.limit - r.used)} übrig`);
    text($('#recordRemaining'), `${Math.max(0, a.limit - a.used)} übrig`);
    const queueLabels={repair:'Unklarheiten & Reparaturen',audit:'Zufällige Kontrollen',reference:'Development-Referenzen',comparison:'Modellvergleiche'};
    for(const option of $('#queueKind').options){const item=r.by_kind?.[option.value];if(item)option.textContent=`${queueLabels[option.value]} · ${item.used}/${item.limit}`;}
    const used = state.count % 5;
    text($('#sessionLabel'), state.paused ? 'Pausiert · du kannst später fortsetzen' : `Sitzung ${used || (state.count ? 5 : 0)} von 5`);
    $('#sessionTrack').style.width = `${used * 20}%`;
  }
  function renderStats() {
    const s = state.status?.counts || {};
    const families = state.status?.families || {};
    const recordings = state.status?.recordings || {};
    const items = [
      ['Familien · Train', families.train ?? 0], ['Familien · Development', families.validation ?? families.dev ?? 0], ['Familien · verschlossener Test', families.test ?? 0], ['Varianten / Aufgaben', Object.values(s).reduce((sum,value)=>sum+Number(value||0),0)], ['Maschinell akzeptiert', s.auto_accepted ?? 0], ['Menschlich akzeptiert', s.human_accepted ?? 0], ['Menschlich korrigiert', state.status?.human_corrected_tasks ?? 0], ['Ungelöst', s.needs_human ?? 0], ['Zurückgestellt', s.deferred ?? 0], ['Ausgeschlossen', s.quarantined ?? s.excluded ?? 0], ['Aufnahmeversuche', budget('recordings').used], ['Nutzbare Aufnahmen', recordings.confirmed ?? 0], ['ASR-Verlust', recordings.asr_unrecoverable ?? 0], ['ASR offen / fehlgeschlagen', (recordings.asr_pending ?? 0)+(recordings.asr_failed ?? 0)]
    ];
    $('#statsGrid').replaceChildren(...items.map(([label, value]) => { const card = document.createElement('div'); card.className = 'stat-card'; const number = document.createElement('strong'); number.textContent = value; const caption = document.createElement('span'); caption.textContent = label; card.append(number, caption); return card; }));
    const r = budget('reviews'), a = budget('recordings');
    text($('#budgetDetail'), `Reviews: ${r.used}/${r.limit} verbraucht · ${Math.max(0, r.limit-r.used)} übrig\nAufnahmen: ${a.used}/${a.limit} verbraucht · ${Math.max(0, a.limit-a.used)} übrig`);
    const blockers = state.status?.blockers || [];
    if (!blockers.length) {
      const openTasks = s.needs_human ?? s.pending_auto ?? 0;
      const activeASR = (recordings.asr_pending ?? 0) + (recordings.asr_failed ?? 0);
      if (openTasks) blockers.push(`${openTasks} Textaufgaben brauchen noch eine begrenzte Prüfung.`);
      if (activeASR) blockers.push(`${activeASR} Aufnahme-ASR-Aufgaben sind offen oder fehlgeschlagen.`);
      if (s.quarantined) blockers.push(`${s.quarantined} Fälle bleiben ausgeschlossen.`);
    }
    $('#blockerList').replaceChildren(...(blockers.length ? blockers : ['Keine gemeldeten Sperren.']).map((item) => { const li = document.createElement('li'); li.textContent = typeof item === 'string' ? item : (item.message || JSON.stringify(item)); return li; }));
  }
  function setView(view) {
    for (const item of ['review','record','progress']) $(`#${item}View`).classList.toggle('hidden', item !== view);
    document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.view === view));
    if (view === 'progress') refresh();
    if (view === 'record') loadRecordingTasks();
  }
  function diff(a, b) {
    const left = a.match(/\s+|[^\s]+/g) || [], right = b.match(/\s+|[^\s]+/g) || [];
    let prefix = 0; while (prefix < left.length && prefix < right.length && left[prefix] === right[prefix]) prefix++;
    let suffix = 0; while (suffix < left.length-prefix && suffix < right.length-prefix && left[left.length-1-suffix] === right[right.length-1-suffix]) suffix++;
    const fragment = document.createDocumentFragment();
    const add = (tag, value) => { if (!value) return; const el = document.createElement(tag); el.textContent = value.join(''); fragment.append(el); };
    add('span', left.slice(0,prefix)); add('del', left.slice(prefix,left.length-suffix)); add('ins', right.slice(prefix,right.length-suffix)); add('span', left.slice(left.length-suffix));
    $('#diffText').replaceChildren(fragment);
  }
  function showCard(card) {
    state.card = card || null;
    state.conflicted=false;
    $('#queueKind').disabled=!!card;
    $('#reviewEmpty').classList.toggle('hidden', !!card);
    $('#reviewCard').classList.toggle('hidden', !card);
    if (!card) { $('#failureRow').classList.add('hidden'); return; }
    text($('#cardKind'), String(card.kind || 'REVIEW').replaceAll('_',' ').toUpperCase());
    text($('#cardLocale'), card.language || card.locale || '');
    text($('#cardQuestion'), card.question || 'Ist das Ziel vollständig und bedeutungstreu?');
    text($('#cardDetail'), [card.split && `Split: ${card.split}`, card.commands && `Modus: ${JSON.stringify(card.commands)}`, card.terminology && `Terminologie: ${JSON.stringify(card.terminology)}`].filter(Boolean).join(' · ') || '');
    text($('#rawText'), card.raw_transcript ?? card.raw_text ?? '');
    const isReference = ['reference','reference_dev','gold_reference'].includes(card.kind);
    const isComparison = ['comparison','model_comparison'].includes(card.kind);
    const baseOutput=card.output ?? (isReference ? (card.raw_transcript ?? card.raw_text ?? '') : '');
    const draftKey=draftStorageKey(card);
    let draft=null;
    try { const saved=localStorage.getItem(draftKey); if(saved) draft=JSON.parse(saved); } catch {}
    const matchingDraft=draft && draft.task_id===card.task_id && draft.input_sha256===card.input_sha256 && draft.revision===card.revision && typeof draft.output==='string';
    $('#outputText').value = matchingDraft ? draft.output : baseOutput;
    $('#outputText').readOnly = isComparison;
    let pendingDecision=null;try{pendingDecision=JSON.parse(localStorage.getItem(`os_review_pending:${card.task_id}:${card.input_sha256}:${card.revision}`)||'null');}catch{}
    state.pendingDecision=Boolean(pendingDecision?.payload);
    const acceptButton=$('[data-action="accept"]');acceptButton.textContent=state.pendingDecision?'Speicherversuch wiederholen':'So übernehmen';
    setDecisionBusy(false);
    diff($('#rawText').textContent, $('#outputText').value);
    const isAssessment = isComparison || card.kind === 'audit';
    $('#failureRow').classList.toggle('hidden', !isAssessment || isComparison);
    $('#meaningAssessment').value='';
    $('#errorCodes').replaceChildren(...(card.error_code_options || []).map((code) => { const label = document.createElement('label'); const input = document.createElement('input'); input.type='checkbox'; input.value=code; label.append(input, document.createTextNode(` ${code}`)); return label; }));
    renderAnswerComparisons(isComparison ? card.answers || [] : []);
    setDecisionBusy(false);
    flash(state.pendingDecision ? 'Eine gespeicherte Entscheidung wartet. Nutze „Speicherversuch wiederholen“; der Request bleibt unverändert.' : matchingDraft ? `Ungespeicherter Entwurf · Revision ${card.revision}` : `Revision ${card.revision} geladen`, Boolean(matchingDraft||state.pendingDecision));
  }
  function draftStorageKey(card){return `os_review_draft:${card.task_id}:${card.input_sha256}:${card.revision}`;}
  function saveDraft(){if(!state.card||$('#outputText').readOnly)return;const draft={task_id:state.card.task_id,input_sha256:state.card.input_sha256,revision:state.card.revision,output:$('#outputText').value,updated_at:new Date().toISOString()};try{localStorage.setItem(draftStorageKey(state.card),JSON.stringify(draft));flash(`Ungespeicherter Entwurf · Revision ${state.card.revision}`);}catch{flash('Der Browser konnte den Entwurf nicht lokal speichern.',true);}}
  function setDecisionBusy(busy){state.decisionBusy=busy;document.querySelectorAll('[data-action]').forEach((button)=>{button.disabled=busy||state.conflicted||(state.pendingDecision&&button.dataset.action!=='accept');});$('#outputText').disabled=busy||state.conflicted||state.pendingDecision;document.querySelectorAll('#failureRow input,#failureRow select,#answerCards input,#answerCards select').forEach((field)=>{field.disabled=busy||state.conflicted||state.pendingDecision;});}
  function renderAnswerComparisons(answers) {
    const section=$('#answerComparisons'); section.classList.toggle('hidden',answers.length===0);
    const cards=answers.slice(0,2).map((answer,index)=>{
      const card=document.createElement('article'); card.className='answer-card';
      const title=document.createElement('h4'); title.textContent=`Antwort ${String.fromCharCode(65+index)}`;
      const output=document.createElement('pre'); output.textContent=answer.output || '';
      const assessmentLabel=document.createElement('label'); assessmentLabel.textContent='Bedeutungsprüfung';
      const assessment=document.createElement('select'); assessment.dataset.answerIndex=index; assessment.className='answer-assessment';
      for(const [value,label] of [['','Bitte bewerten'],['passed','Kein Bedeutungsfehler'],['failure','Bedeutungsfehler']]){const option=document.createElement('option');option.value=value;option.textContent=label;assessment.append(option);}
      const codes=document.createElement('div'); codes.className='check-list';
      for(const code of (state.card?.error_code_options || [])) {const label=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.dataset.answerIndex=index;input.value=code;label.append(input,document.createTextNode(` ${code}`));codes.append(label);}
      card.append(title,output,assessmentLabel,assessment,codes); return card;
    });
    $('#answerCards').replaceChildren(...cards);
  }
  async function nextCard() {
    if (state.paused) return;
    try { const result = await api('/api/next-card', {kind:$('#queueKind').value}); const card = result.card || result; showCard(card?.task_id ? card : null); if (!card?.task_id) { text($('#emptyTitle'),'In dieser Aufgabenart sind gerade keine Karten verfügbar.'); text($('#emptyCopy'),'Du kannst eine andere Aufgabenart wählen oder später erneut nachsehen.'); text($('#refreshReview'),'Erneut nachsehen'); } if (result.budgets) state.status.budgets = result.budgets; renderBudgets(); }
    catch (error) { flash(error.message, true); }
  }
  async function decide(action) {
    if (!state.card||state.decisionBusy||state.conflicted) return;
    const pendingKey=`os_review_pending:${state.card.task_id}:${state.card.input_sha256}:${state.card.revision}`;
    let pending=null;
    try{pending=JSON.parse(localStorage.getItem(pendingKey)||'null');}catch{}
    if(pending){action=pending.payload.action;}
    let payload=pending?.payload;
    try {
      if(!payload){
        const answer_reviews=[...$('#answerCards').children].map((card,index)=>({answer_index:index,semantic_failure:card.querySelector('.answer-assessment').value===''?null:card.querySelector('.answer-assessment').value==='failure',error_codes:[...card.querySelectorAll('.check-list input:checked')].map((input)=>input.value)}));
        if (['accept','edit'].includes(action) && answer_reviews.some((review)=>review.semantic_failure===null)) { flash('Bitte bewerte beide Antworten ausdrücklich.',true); return; }
        const semantic_failure=state.card.kind==='audit'?($('#meaningAssessment').value===''?null:$('#meaningAssessment').value==='failure'):(answer_reviews.length?answer_reviews.some((review)=>review.semantic_failure===true):null);
        if (['accept','edit'].includes(action) && state.card.kind==='audit' && semantic_failure===null) { flash('Bitte wähle: kein Bedeutungsfehler oder Bedeutungsfehler.',true); return; }
        payload={task_id:state.card.task_id,expected_revision:state.card.revision,input_sha256:state.card.input_sha256,action,output:$('#outputText').value,request_id:id(),semantic_failure,error_codes:[...$('#errorCodes').querySelectorAll('input:checked')].map((x)=>x.value),answer_reviews};
      }
      localStorage.setItem(pendingKey,JSON.stringify({payload,saved_at:new Date().toISOString()}));
      state.pendingDecision=true;
      setDecisionBusy(true);
      const result = await api('/api/decision', payload);
      localStorage.removeItem(pendingKey);
      localStorage.removeItem(draftStorageKey(state.card));
      state.pendingDecision=false;
      if (action !== 'undo') { state.count++; localStorage.setItem('os_review_session_count',String(state.count)); }
      text($('#saveState'), action === 'defer' ? 'Zurückgestellt' : action === 'exclude' ? 'Ausgeschlossen' : action === 'undo' ? 'Rückgängig dokumentiert' : 'Gespeichert');
      if (result.budgets) state.status.budgets = result.budgets;
      await refresh();
      if (action !== 'undo' && state.count % 5 === 0) { state.paused = true; localStorage.setItem('os_review_paused','true'); $('#pauseButton').textContent='Fortsetzen'; showCard(null); text($('#emptyTitle'),'Dein Fünferblock ist erledigt.'); text($('#emptyCopy'),'Du hast fünf Karten bearbeitet. Nimm dir eine Pause oder setze mit einem neuen Block fort.'); text($('#refreshReview'),'Neuen Fünferblock beginnen'); }
      else await nextCard();
    } catch (error) {
      if(error.code==='revision_conflict'){localStorage.removeItem(pendingKey);state.conflicted=true;setDecisionBusy(false);flash('Diese Karte wurde inzwischen geändert. Der ausstehende Entscheidungsversuch wurde verworfen. Bitte lade die Seite neu, bevor du fortsetzt.',true);}
      else {state.pendingDecision=true;flash(pending?'Speichern ist noch offen. Klicke auf „Speicherversuch wiederholen“, um denselben Request sicher zu senden.':`Speichern fehlgeschlagen: ${error.message}. Derselbe Request bleibt zum sicheren Wiederholen vorgemerkt.`,true);}
    } finally { setDecisionBusy(false); }
  }
  async function refresh() { try { await loadStatus(true); } catch (error) { flash(error.message, true); } }
  async function loadRecordingTasks(preferNext=false) {
    try {
      const result = await api('/api/recording-tasks'); state.recordingTasks = Array.isArray(result) ? result : (result.tasks || []); renderRecordingTasks();
      let resume=null;try{resume=JSON.parse(localStorage.getItem('os_active_recording')||'null');}catch{}
      const task=(!preferNext&&state.recordingTasks.find(item=>item.task_id===resume?.task_id))||(!preferNext&&state.recordingTasks.find(item=>item.task_id===state.task?.task_id))||(!preferNext&&state.recordingTasks.find(item=>['needs_confirmation','asr_pending','asr_failed','recording'].includes(item.recording_status)))||state.recordingTasks.find(item=>!item.recording_status)||state.recordingTasks[0];
      if(task)await selectRecordingTask(task);
    }
    catch (error) { text($('#recordStatus'), error.message); }
  }
  function renderRecordingTasks() {
    const select=$('#recordTaskSelect');select.replaceChildren();
    const labels={needs_confirmation:'Ziele prüfen',asr_pending:'Transkription läuft',asr_failed:'Fehler – Audio vorhanden',confirmed:'Gespeichert',recording:'Audio noch offen',deleted:'Gelöscht',asr_unrecoverable:'ASR-Verlust',discarded:'Verworfen – neu aufnehmen'};
    state.recordingTasks.forEach((task,index)=>{const option=document.createElement('option');option.value=task.task_id;option.textContent=`${index+1}. ${task.language} · ${task.category||task.task_id}${task.recording_status?' · '+labels[task.recording_status]:''}`;select.append(option);});
    select.disabled=!state.recordingTasks.length||state.recorder?.state==='recording';
  }
  async function selectRecordingTask(task) {
    if(state.recorder?.state==='recording'||state.uploading)return;
    if(state.pollTimer){clearTimeout(state.pollTimer);state.pollTimer=null;}
    state.task=task; text($('#recordLocale'), `${task.language || task.locale || ''} · ${task.language_name || ''}`);
    $('#recordTaskSelect').value=task.task_id;
    const script=task.script||task.speaking_text;
    text($('#recordHelp'),script?'2. Sprechtext natürlich einsprechen und stoppen. 3. Lokale Transkription abwarten, bereinigte Ziele prüfen und speichern. Fehlende Fakten nicht aus der Vorlage ergänzen.':'2. Eine kurze Nachricht oder Alltagssituation in eigenen Worten erzählen und stoppen. 3. Transkription abwarten, Ziele prüfen und speichern.');
    text($('#recordPromptLabel'),script?'SPRECHTEXT · DIESEN TEXT EINSPRECHEN':'FREIE SITUATION · IN EIGENEN WORTEN');
    text($('#recordPrompt'), script?script.replace(/^(?:Bitte lies|Dictate|Sprich)[^:]*:\s*/i,''):(task.prompt||'Erzähle frei von einer kurzen Aufgabe oder Nachricht.'));
    text($('#recordTarget'), task.intended_output || task.target || '');
    $('#recordTarget').parentElement.open=false;
    $('#recordTarget').parentElement.classList.toggle('hidden',!task.intended_output&&!task.target);
    $('#asrArea').classList.add('hidden');$('#retryASR').classList.add('hidden');$('#playback').classList.add('hidden');
    state.recordingId=task.latest_recording_id||null;
    $('#recordStart').disabled = !task.task_id || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined';
    $('#recordStatus').textContent = $('#recordStart').disabled ? 'Aufnahme ist in diesem Browser nicht verfügbar.' : 'Aufgabe gewählt · Mikrofon wird erst beim Start aktiviert';
    if(state.recordingId&&['needs_confirmation','asr_pending','asr_failed','recording'].includes(task.recording_status)){
      localStorage.setItem('os_active_recording',JSON.stringify({recording_id:state.recordingId,task_id:task.task_id}));
      $('#recordStart').disabled=true;await pollRecording(true);
    }
  }
  $('#recordTaskSelect').addEventListener('change',()=>{const task=state.recordingTasks.find(item=>item.task_id===$('#recordTaskSelect').value);if(task)selectRecordingTask(task);});
  async function enumerateMics() {
    const select=$('#micSelect');
    try { const devices=await navigator.mediaDevices.enumerateDevices(); const mics=devices.filter(d=>d.kind==='audioinput'); select.replaceChildren(...mics.map((device,index)=>{const option=document.createElement('option');option.value=device.deviceId;option.textContent=device.label||`Mikrofon ${index+1}`;return option;})); if(!mics.length){const option=document.createElement('option');option.textContent='Kein Mikrofon gefunden';select.append(option);} }
    catch { select.innerHTML='<option>Mikrofonliste nicht verfügbar</option>'; }
  }
  async function startRecording() {
    if (!state.task||state.recorder?.state==='recording'||state.uploading) return;
    state.recordingEngine=$('#asrEngine').value;
    try {
      if(!state.statusReady)await loadStatus();
      state.stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:$('#micSelect').value ? {exact:$('#micSelect').value}:undefined,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
      state.audioSettings=state.stream.getAudioTracks()[0]?.getSettings?.()||{};
      startLevelMeter();
      const types=['audio/webm;codecs=opus','audio/ogg;codecs=opus','audio/webm','audio/mp4']; const mime=types.find((t)=>MediaRecorder.isTypeSupported(t));
      if (!mime) throw new Error('Dieser Browser bietet kein unterstütztes lokales Audioformat.');
      const started=await api('/api/recording/start',{task_id:state.task.task_id,request_id:id(),consent_local:true}); state.recordingId=started.recording_id;localStorage.setItem('os_active_recording',JSON.stringify({recording_id:state.recordingId,task_id:state.task.task_id,engine:$('#asrEngine').value}));
      state.recordingFailed=false;state.recorder=new MediaRecorder(state.stream,{mimeType:mime}); state.chunks=[]; state.recorder.ondataavailable=(e)=>{if(e.data.size)state.chunks.push(e.data);}; state.recorder.onerror=(event)=>{state.recordingFailed=true;text($('#recordStatus'),event.error?.message||'Aufnahme fehlgeschlagen. Der Versuch bleibt gezählt; es erfolgt keine automatische Wiederholung.');stopRecording();}; state.recorder.onstop=finishRecording; state.recorder.start();
      state.startedAt=Date.now(); $('#recordStart').disabled=true; $('#recordStop').disabled=false; text($('#recordStatus'),'Aufnahme läuft · nur lokal');
      $('#recordTaskSelect').disabled=true;$('#asrEngine').disabled=true;
      state.timer=setInterval(()=>{const seconds=Math.floor((Date.now()-state.startedAt)/1000); text($('#recordDuration'),`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`);},250);
    } catch(error) { state.stream?.getTracks().forEach(t=>t.stop()); state.stream=null; await stopLevelMeter(); text($('#recordStatus'),error.message); if(state.recordingId){$('#asrArea').classList.remove('hidden');$('#confirmRecord').disabled=true;pollRecording(true);} }
  }
  function startLevelMeter(){
    const AudioContextClass=window.AudioContext||window.webkitAudioContext;
    if(!AudioContextClass||!state.stream)return;
    try{
      state.audioContext=new AudioContextClass();state.analyser=state.audioContext.createAnalyser();state.analyser.fftSize=256;
      state.audioContext.createMediaStreamSource(state.stream).connect(state.analyser);
      const samples=new Uint8Array(state.analyser.fftSize);
      const draw=()=>{if(!state.analyser)return;state.analyser.getByteTimeDomainData(samples);let sum=0;for(const sample of samples){const normalized=(sample-128)/128;sum+=normalized*normalized;}const level=Math.min(100,Math.round(Math.sqrt(sum/samples.length)*260));$('#audioLevel').style.width=`${level}%`;state.levelFrame=requestAnimationFrame(draw);};draw();
    }catch{ $('#audioLevel').style.width='0%'; }
  }
  async function stopLevelMeter(){if(state.levelFrame)cancelAnimationFrame(state.levelFrame);state.levelFrame=null;state.analyser=null;$('#audioLevel').style.width='0%';if(state.audioContext){await state.audioContext.close().catch(()=>{});state.audioContext=null;}}
  function stopRecording() { if(state.recorder?.state==='recording')state.recorder.stop(); state.stream?.getTracks().forEach(t=>t.stop()); state.stream=null; stopLevelMeter(); clearInterval(state.timer); $('#recordStop').disabled=true; }
  async function finishRecording() {
    if(state.recordingFailed){$('#recordTaskSelect').disabled=false;$('#asrEngine').disabled=false;state.stream?.getTracks().forEach((track)=>track.stop());state.stream=null;await stopLevelMeter();await pollRecording(true);return;}
    state.uploading=true;text($('#recordStatus'),'Audio wird lokal gespeichert …');
    const blob=new Blob(state.chunks,{type:state.recorder.mimeType}); $('#playback').src=URL.createObjectURL(blob); $('#playback').classList.remove('hidden');
    try {
      const base64=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(reader.error);reader.readAsDataURL(blob);});
      const settings=state.audioSettings;
      const browser_metadata={sample_rate:settings.sampleRate??null,channel_count:settings.channelCount??null,echo_cancellation:settings.echoCancellation??null,noise_suppression:settings.noiseSuppression??null,auto_gain_control:settings.autoGainControl??null,device_id:settings.deviceId??null,recorder_mime:blob.type};
      const result=await api('/api/recording/upload',{recording_id:state.recordingId,request_id:id(),mime:blob.type,base64,engine:state.recordingEngine,language:state.task?.language || 'auto',browser_metadata});
      state.recordingId=result.recording_id || state.recordingId;
      $('#asrArea').classList.remove('hidden'); $('#confirmRecord').disabled=true; text($('#recordStatus'),'Audio lokal gespeichert · lokale Transkription läuft …');
      await pollRecording();
    } catch(error) { text($('#recordStatus'),`Audio konnte nicht gespeichert werden: ${error.message}`); await pollRecording(true); }
    state.uploading=false;$('#recordTaskSelect').disabled=false;$('#asrEngine').disabled=false;await refresh();
  }
  async function pollRecording(resumed=false) {
    if(!state.recordingId||state.polling||state.pollTimer)return;
    state.polling=true;
    try {
      const requestedId=state.recordingId;const status=await api(`/api/recording/${encodeURIComponent(requestedId)}`);
      if(requestedId!==state.recordingId)return;
      $('#retryASR').classList.toggle('hidden',status.status!=='asr_failed');
      if(status.status==='asr_pending') { showSavedAudio();$('#asrArea').classList.remove('hidden');$('#recordStart').disabled=true;text($('#recordStatus'),'Lokale Transkription läuft …'); state.pollTimer=setTimeout(()=>{state.pollTimer=null;pollRecording(resumed);},1200); return; }
      if(status.status==='needs_confirmation' || status.variants?.length) { showSavedAudio();$('#asrArea').classList.remove('hidden');$('#recordStart').disabled=true;renderASRVariants(status); text($('#recordStatus'),'Rohtranskription bereit · Ziele getrennt prüfen'); return; }
      if(status.status==='recording'){ $('#asrArea').classList.remove('hidden');$('#confirmRecord').disabled=true;$('#recordStart').disabled=true;text($('#recordStatus'),resumed?'Vor dem Neuladen noch kein Audioclip gespeichert. Der Aufnahmeversuch zählt bereits; lösche diesen offenen Versuch, wenn du neu aufnehmen möchtest.':'Audio konnte nicht gespeichert werden. Der Aufnahmeversuch zählt bereits; erneuter Versuch ist manuell möglich.');return; }
      if(['asr_failed','asr_unrecoverable'].includes(status.status)){ showSavedAudio();$('#asrArea').classList.remove('hidden');$('#confirmRecord').disabled=true;$('#recordStart').disabled=true;text($('#recordStatus'),`Transkription fehlgeschlagen: ${status.error||'Kein nutzbarer Rohtext'}. Das Audio bleibt gespeichert; die Transkription kann ohne neuen Aufnahmeversuch erneut starten.`);return; }
      text($('#recordStatus'),status.error || `Aufnahmestatus: ${status.status}`);
      if(['confirmed','deleted'].includes(status.status))localStorage.removeItem('os_active_recording');
    } catch(error) { text($('#recordStatus'),error.message);if(error.status===404){state.recordingId=null;localStorage.removeItem('os_active_recording');} }
    finally { state.polling=false; }
  }
  function showSavedAudio(){if(!state.recordingId)return;$('#playback').src=`/api/audio/${encodeURIComponent(state.recordingId)}`;$('#playback').classList.remove('hidden');}
  function renderASRVariants(status) {
    const variants=status.variants || (status.raw_transcript ? [{engine:'nemotron',transcript:status.raw_transcript}] : []);
    const errors=status.errors || [];
    $('#asrVariants').replaceChildren(); $('#asrTargets').replaceChildren();
    for(const variant of variants) {
      const engine=variant.engine || 'ASR';
      const title=document.createElement('label'); title.textContent=`${engine.toUpperCase()} · UNVERÄNDERLICHE ROHTRANSKRIPTION`;
      const raw=document.createElement('textarea'); raw.rows=5; raw.readOnly=true; raw.value=variant.raw_transcript || variant.transcript || '';
      $('#asrVariants').append(title,raw);
      const targetLabel=document.createElement('label'); targetLabel.textContent=`BEREINIGTES ZIEL · ${engine.toUpperCase()}`;
      const target=document.createElement('textarea'); target.rows=4; target.dataset.engine=engine;const draftKey=`os_asr_draft:${state.recordingId}:${engine}:${variant.raw_sha256||''}`;target.value=localStorage.getItem(draftKey)??raw.value;target.addEventListener('input',()=>{try{localStorage.setItem(draftKey,target.value);}catch{text($('#recordStatus'),'Entwurf konnte nicht im Browser gespeichert werden.');}});
      const lossLabel=document.createElement('label'); const loss=document.createElement('input'); loss.type='checkbox'; loss.dataset.engine=engine; loss.className='asr-loss';loss.checked=localStorage.getItem(draftKey+':loss')==='true';loss.addEventListener('change',()=>localStorage.setItem(draftKey+':loss',String(loss.checked))); lossLabel.append(loss,document.createTextNode(` ${engine}: ASR-Verlust · im Rohtext fehlt nicht rekonstruierbarer Inhalt`));
      $('#asrTargets').append(targetLabel,target,lossLabel);
    }
    for(const failure of errors) { const p=document.createElement('p');p.className='notice warning-notice';p.textContent=`${failure.engine}: ${failure.error || 'Engine nicht verfügbar'}`;$('#asrVariants').append(p); }
    $('#confirmRecord').disabled=variants.length===0;
  }
  function clearRecordingDrafts(rid){for(const key of Object.keys(localStorage)){if(key.startsWith(`os_asr_draft:${rid}:`))localStorage.removeItem(key);}}
  $('#retryASR').addEventListener('click',async()=>{
    if(!state.recordingId)return;$('#retryASR').disabled=true;
    try{await api('/api/recording/retry',{recording_id:state.recordingId,request_id:id()});text($('#recordStatus'),'Gespeichertes Audio wird erneut transkribiert · kein neuer Aufnahmeversuch');await pollRecording();}
    catch(error){text($('#recordStatus'),error.message);}finally{$('#retryASR').disabled=false;}
  });
  $('#confirmRecord').addEventListener('click',async()=>{
    try { const outputs={}; for(const target of $('#asrTargets').querySelectorAll('textarea[data-engine]'))outputs[target.dataset.engine]=target.value; const unrecoverable_engines=[...$('#asrTargets').querySelectorAll('.asr-loss:checked')].map((input)=>input.dataset.engine); const result=await api('/api/recording/confirm',{recording_id:state.recordingId,outputs,unrecoverable_engines,request_id:id()}); clearRecordingDrafts(state.recordingId);localStorage.removeItem('os_active_recording'); text($('#recordStatus'),result.message||'Aufnahmeziele gespeichert'); $('#asrArea').classList.add('hidden'); $('#recordStart').disabled=false;await refresh();await loadRecordingTasks(true); }
    catch(error){text($('#recordStatus'),error.message);}
  });
  $('#skipRecord').addEventListener('click',async()=>{try{await api('/api/recording/discard',{recording_id:state.recordingId,request_id:id()});localStorage.removeItem('os_active_recording');await refresh();await loadRecordingTasks(true);text($('#recordStatus'),'Vorherige Aufnahme verworfen. Audio bleibt lokal erhalten, der Versuch bleibt gezählt.');}catch(error){text($('#recordStatus'),error.message);}});
  $('#deleteRecording').addEventListener('click',async()=>{
    $('#dialogMessage').textContent='Audio und abgeleitete Aufnahmeinhalte dieser Aufgabe löschen? Dieser lokale Clip wird danach nicht wiederhergestellt.';
    const dialog=$('#confirmDialog');dialog.showModal();const result=await new Promise(resolve=>dialog.addEventListener('close',()=>resolve(dialog.returnValue),{once:true}));
    if(result!=='confirm')return;
    try{await api('/api/recording/delete',{recording_id:state.recordingId,request_id:id()});clearRecordingDrafts(state.recordingId);localStorage.removeItem('os_active_recording');$('#asrArea').classList.add('hidden');$('#playback').removeAttribute('src');$('#playback').classList.add('hidden');$('#recordStart').disabled=false;state.recordingId=null;text($('#recordStatus'),'Audio und abgeleitete Aufnahme gelöscht. Der Versuch bleibt gezählt.');await refresh();}
    catch(error){text($('#recordStatus'),error.message);}
  });
  document.querySelectorAll('.tab').forEach((tab)=>tab.addEventListener('click',()=>setView(tab.dataset.view)));
  document.querySelectorAll('[data-action]').forEach((button)=>button.addEventListener('click',()=>decide(button.dataset.action)));
  $('#outputText').addEventListener('input',()=>{diff($('#rawText').textContent,$('#outputText').value);saveDraft();});
  function beginSession(){if(state.paused){state.paused=false;localStorage.setItem('os_review_paused','false');if(state.count%5===0){state.count=0;localStorage.setItem('os_review_session_count','0');}}$('#pauseButton').textContent='Für heute fertig';text($('#emptyTitle'),'Bereit für einen kurzen Block?');text($('#emptyCopy'),'Starte bewusst bis zu fünf Karten. Bereits geöffnete Karten werden beim Fortsetzen wieder aufgenommen.');text($('#refreshReview'),'Fünferblock beginnen / fortsetzen');nextCard();}
  $('#refreshReview').addEventListener('click',beginSession);
  $('#queueKind').addEventListener('change',()=>{if(!state.paused)nextCard();});
  $('#pauseButton').addEventListener('click',()=>{state.paused=!state.paused;localStorage.setItem('os_review_paused',String(state.paused));$('#pauseButton').textContent=state.paused?'Fortsetzen':'Für heute fertig';renderBudgets();if(state.paused){showCard(null);text($('#emptyTitle'),'Dein Fünferblock ist pausiert.');text($('#emptyCopy'),'Deine aktuelle Karte bleibt offen. Du kannst später an dieser Stelle fortsetzen.');text($('#refreshReview'),'Fünferblock fortsetzen');}else beginSession();});
  $('#recordStart').addEventListener('click',startRecording); $('#recordStop').addEventListener('click',stopRecording);
  $('#extendButton').addEventListener('click',async()=>{
    $('#dialogMessage').textContent='Möchtest du dein Review-Budget und Aufnahmebudget erweitern? Die Gesamtgrenzen bleiben bei 100 Reviews und 20 Versuchen.';
    const dialog=$('#confirmDialog');dialog.showModal();const result=await new Promise(resolve=>dialog.addEventListener('close',()=>resolve(dialog.returnValue),{once:true}));
    if(result==='confirm')try{const payload=await api('/api/extend',{confirm:true});getStatus({csrf_token:state.csrf,status:payload.dashboard||payload});}catch(error){text($('#exportNotice'),error.message);$('#exportNotice').classList.remove('hidden');}
  });
  $('#exportButton').addEventListener('click',async()=>{try{const result=await api('/api/export',{});text($('#exportNotice'),result.message||`Snapshot vorbereitet: ${result.path||result.snapshot_id||'lokal'}. Export startet kein Training.`);$('#exportNotice').classList.remove('hidden');}catch(error){text($('#exportNotice'),error.message);$('#exportNotice').classList.remove('hidden');}});
  enumerateMics(); if(navigator.mediaDevices?.addEventListener)navigator.mediaDevices.addEventListener('devicechange',enumerateMics);
  $('#pauseButton').textContent=state.paused?'Fortsetzen':'Für heute fertig';
  if(state.paused){text($('#emptyTitle'),'Dein Fünferblock ist pausiert.');text($('#emptyCopy'),'Deine aktuelle Karte bleibt offen. Du kannst später an dieser Stelle fortsetzen.');text($('#refreshReview'),'Fünferblock fortsetzen');}
  refresh();
})();
