import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { ArrowDown, ArrowRight, ArrowUpRight, BookOpen, Braces, Check, CheckCheck, ChevronDown, CircleHelp, Clipboard, Copy, Cpu, FileText, GitBranch, Layers3, LoaderCircle, Microscope, Play, RefreshCw, ScanText, ShieldCheck, Terminal, Trash2, TriangleAlert } from 'lucide-react'
import { request, type Analysis, type ModelStatus, type Rule } from './api'

const stages = [
  { id: 'M1', name: 'Lexer', description: 'Text → tokens', icon: ScanText },
  { id: 'M2', name: 'Symbol table', description: 'Tokens → vocabulary', icon: Braces },
  { id: 'M3', name: 'Parser', description: 'Words → structure', icon: GitBranch },
  { id: 'M4', name: 'Features', description: 'Structure → signals', icon: Layers3 },
  { id: 'M5', name: 'Scoring', description: 'Signals → verdict', icon: Microscope },
]
const format = (value: number, places = 3) => value.toLocaleString('en-IN', { maximumFractionDigits: places, minimumFractionDigits: places })
const percent = (value: number) => `${format(value * 100, 1)}%`
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please retry.'

function Brand({ small = false }: { small?: boolean }) {
  return <a className={`brand ${small ? 'brand-small' : ''}`} href="#workspace" aria-label="AI Content Breaker home">
    <span className="brand-mark"><GitBranch size={24} strokeWidth={1.6} /></span>
    <span>content<span className="brand-second">breaker<span className="brand-dot">.</span></span></span>
  </a>
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return <div className="metric"><dt>{label}</dt><dd>{value}</dd></div>
}

function Score({ label, value, verdict, abstain }: { label: string; value: number | null; verdict: string | null; abstain: boolean }) {
  const readable = abstain ? 'Insufficient text' : verdict === 'ai' ? 'Verdict: AI' : verdict === 'human' ? 'Verdict: Human' : 'Model unavailable'
  return <div className="score">
    <div className="score-heading"><span>{label}</span><span className="score-value">{abstain || value === null ? '—' : percent(value)}</span></div>
    <div className="score-track" aria-hidden="true"><span style={{ width: `${!abstain && value !== null ? value * 100 : 0}%` }} /></div>
    <div className="score-caption"><span>{readable}</span><span>{abstain || value === null ? 'No verdict' : 'AI score'}</span></div>
  </div>
}

function RuleTable({ rules }: { rules: Rule[] }) {
  return <div className="table-scroll"><table><thead><tr><th>Rule</th><th>Value</th><th>Threshold</th><th>Weight</th></tr></thead><tbody>{rules.map(([name, value, threshold, weight]) => <tr key={name}><td>{name}</td><td>{format(value, 4)}</td><td>{format(threshold, 4)}</td><td>{weight > 0 ? '+' : ''}{format(weight, 1)}</td></tr>)}</tbody></table></div>
}

function PipelineReport({ result }: { result: Analysis }) {
  const { lexer, symbols, parser, features, rules } = result
  return <div className="pipeline-report">
    {stages.map((stage, index) => <details className="stage-detail" id={`stage-${stage.id}`} key={stage.id} open>
      <summary><span className="stage-code">{stage.id}</span><span className="stage-title">{stage.name}</span><span className="stage-summary">{[`${result.summary.n_tokens} tokens`, `${symbols.unique_count} words`, `${parser.n_sentences} sentences`, `${Object.keys(features.all).length} signals`, `${rules.fired.length} rules fired`][index]}</span><ChevronDown size={16} className="disclosure-chevron" /></summary>
      <div className="stage-body">
        {index === 0 && <><div className="token-counts">{Object.entries(lexer.counts).map(([name, count]) => <div key={name}><span className={`token-dot token-${name.toLowerCase()}`} />{name.toLowerCase().replace('_', ' ')}<strong>{count}</strong></div>)}</div><p className="stage-note">The lexer separates the passage into typed tokens using maximal munch.</p><details className="sub-detail"><summary>Inspect token stream <span>{lexer.tokens.length}</span></summary><div className="token-stream">{lexer.tokens.map((token, i) => <span key={i} className={`token token-${token.type.toLowerCase()}`} title={`${token.type} · position ${token.idx}`}>{token.value}<small>{token.type.toLowerCase()}</small></span>)}</div></details></>}
        {index === 1 && <><p className="stage-note">Most frequent words, with their occurrences in the passage.</p><div className="word-list">{symbols.top_words.map(([word, count]) => <div key={word}><span>{word}</span><span className="word-bar"><i style={{ width: `${count / Math.max(1, symbols.top_words[0]?.[1] || 1) * 100}%` }} /></span><strong>{count}</strong></div>)}</div></>}
        {index === 2 && <><dl className="metric-grid parser-metrics"><Metric label="Parse success" value={percent(parser.parse_success_rate)} /><Metric label="Mean coverage" value={percent(parser.mean_coverage)} /><Metric label="Mean depth" value={format(parser.mean_depth, 2)} /><Metric label="Mean branching" value={format(parser.mean_branching, 2)} /><Metric label="Mean nodes" value={format(parser.mean_nodes, 1)} /><Metric label="POS tagger" value={parser.tagger} /></dl><div className="parser-patterns"><div><h4>Sentence structures</h4>{Object.entries(parser.structure_counts).filter(([, v]) => v > 0).map(([name, count]) => <span className="pattern" key={name}>{name}<b>{count}</b></span>)}</div><div><h4>POS templates</h4>{Object.entries(parser.template_counts).filter(([, v]) => v > 0).map(([name, count]) => <span className="pattern" key={name}>{name}<b>{count}</b></span>)}{!Object.values(parser.template_counts).some(v => v > 0) && <p className="stage-note">No templates matched.</p>}</div></div><details className="sub-detail"><summary>First sentence: tags & parse tree</summary><div className="tree-container"><p className="tag-list">{parser.sentences[0]?.tags.join(' ')}</p><pre>{parser.first_tree || '(no tree: parse failed)'}</pre></div></details></>}
        {index === 3 && <><div className="feature-list">{Object.entries(features.selected).map(([name, value]) => <div key={name}><code>{name}</code><span>{format(value, 4)}</span></div>)}</div><details className="sub-detail"><summary>Inspect all features <span>{Object.keys(features.all).length}</span></summary><div className="feature-list all-features">{Object.entries(features.all).map(([name, value]) => <div key={name}><code>{name}</code><span>{format(value, 4)}</span></div>)}</div></details></>}
        {index === 4 && <><dl className="metric-grid"><Metric label="Rule total" value={`${rules.total > 0 ? '+' : ''}${format(rules.total, 1)}`} /><Metric label="Maximum ±" value={format(rules.max_total, 1)} /><Metric label="Rules fired / missed" value={`${rules.fired.length} / ${rules.missed.length}`} /></dl><h4 className="rule-label">Fired rules</h4>{rules.fired.length ? <RuleTable rules={rules.fired} /> : <p className="stage-note">No rules fired for this passage.</p>}<p className="stage-note">M5a combines inspectable rules. M5b scores the same feature vector with the trained classifier. The cutoff for both is 0.5.</p>{result.summary.model_note && <p className="inline-warning"><TriangleAlert size={16} />{result.summary.model_note}</p>}</>}
      </div>
    </details>)}
  </div>
}

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<Analysis | null>(null)
  const [analyzedText, setAnalyzedText] = useState('')
  const [model, setModel] = useState<ModelStatus | null>(null)
  const [modelError, setModelError] = useState('')
  const [modelFeedback, setModelFeedback] = useState('')
  const [checking, setChecking] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [sampleLoading, setSampleLoading] = useState(false)
  const [error, setError] = useState('')
  const [view, setView] = useState<'pipeline' | 'console'>('pipeline')
  const [copied, setCopied] = useState(false)
  const input = useRef<HTMLTextAreaElement>(null)
  const resultHeading = useRef<HTMLHeadingElement>(null)
  const mounted = useRef(true)
  const modelBusy = useRef(false)
  const copyTimer = useRef<number | undefined>(undefined)
  const guide = useRef<HTMLDetailsElement>(null)
  const wordCount = text.split(/\s+/u).filter(piece => /\p{L}/u.test(piece)).length
  const minWords = model?.min_words || 50
  const maxCharacters = model?.max_characters || 50000
  const training = model?.training.running || false
  const stale = !!result && text !== analyzedText

  useEffect(() => {
    mounted.current = true
    request<ModelStatus>('/api/model').then(value => { if (mounted.current) setModel(value) }).catch(error => { if (mounted.current) setModelError(message(error)) })
    return () => { mounted.current = false; window.clearTimeout(copyTimer.current) }
  }, [])

  useEffect(() => {
    if (!training) return
    let cancelled = false
    let timer: number
    const poll = async () => {
      try {
        const status = await request<ModelStatus>('/api/model')
        if (cancelled) return
        setModel(status)
        setModelError('')
        if (status.training.running) timer = window.setTimeout(poll, 1800)
        else setModelFeedback(status.ready ? 'Training complete. The model is ready.' : '')
      } catch (error) {
        if (cancelled) return
        setModelError(message(error))
        timer = window.setTimeout(poll, 5000)
      }
    }
    timer = window.setTimeout(poll, 1800)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [training])

  async function checkModel() {
    if (modelBusy.current || training) return
    modelBusy.current = true
    setChecking(true); setModelError(''); setModelFeedback('')
    try {
      let status = await request<ModelStatus>('/api/model')
      if (!status.ready && !status.training.running) status = await request<ModelStatus>('/api/model/train', { method: 'POST', body: '{}' })
      if (!mounted.current) return
      setModel(status)
      setModelFeedback(status.ready ? 'Verified. The trained model is ready.' : status.message)
    } catch (error) { if (mounted.current) setModelError(message(error)) }
    finally { modelBusy.current = false; if (mounted.current) setChecking(false) }
  }

  async function analyze(event: FormEvent) {
    event.preventDefault()
    if (analyzing) return
    if (!text.trim()) { setError('Enter a passage to analyze, or load the sample text.'); input.current?.focus(); return }
    if (text.length > maxCharacters) { setError(`Keep the passage under ${maxCharacters.toLocaleString('en-IN')} characters.`); input.current?.focus(); return }
    const submitted = text
    setAnalyzing(true); setError(''); setCopied(false)
    try {
      const analysis = await request<Analysis>('/api/analyze', { method: 'POST', body: JSON.stringify({ text: submitted }) }, 120000)
      if (!mounted.current) return
      setResult(analysis); setAnalyzedText(submitted)
      window.requestAnimationFrame(() => resultHeading.current?.focus({ preventScroll: true }))
    } catch (error) { if (mounted.current) setError(message(error)) }
    finally { if (mounted.current) setAnalyzing(false) }
  }

  async function loadSample() {
    setSampleLoading(true); setError('')
    try { const sample = await request<{ text: string }>('/api/sample'); if (mounted.current) { setText(sample.text); input.current?.focus() } }
    catch (error) { if (mounted.current) setError(message(error)) }
    finally { if (mounted.current) setSampleLoading(false) }
  }

  async function copyReport() {
    try {
      await navigator.clipboard.writeText(result!.console_output)
      setCopied(true)
      window.clearTimeout(copyTimer.current)
      copyTimer.current = window.setTimeout(() => setCopied(false), 2500)
    } catch { setError('Clipboard access is unavailable. Select and copy the text in the console view.') }
  }

  return <div className="app-shell">
    <a className="skip-link" href="#workspace">Skip to analysis</a>
    <aside className="sidebar">
      <Brand />
      <div className="sidebar-caption">AI text detection</div>
      <nav aria-label="Workbench"><a className="nav-link nav-active" href="#workspace"><ScanText size={18} />Text analysis<ArrowUpRight size={14} /></a><a className="nav-link" href="#pipeline-guide" onClick={() => { if (guide.current) guide.current.open = true }}><BookOpen size={18} />Pipeline guide</a></nav>
      <div className="sidebar-bottom">
        <div className="model-panel">
          <div className="model-label"><Cpu size={17} /><span>Local model</span><span className={`status-dot ${model?.ready ? 'ready' : training ? 'working' : ''}`} /></div>
          <div className="model-state">{training ? 'Training in progress' : model?.ready ? 'Trained & ready' : model ? 'Setup needed' : modelError ? 'Server unavailable' : 'Checking model…'}</div>
          <p>{model?.ready ? 'The classifier is ready to analyze your text.' : 'Check the saved model. If needed, training starts automatically.'}</p>
          <button className="button button-outline model-button" onClick={checkModel} disabled={checking || training}>{checking || training ? <LoaderCircle size={15} className="spin" /> : <RefreshCw size={15} />}{training ? 'Training model' : checking ? 'Checking model' : model?.ready ? 'Check model' : 'Check & train model'}</button>
          <div className="model-feedback" role="status">{modelFeedback && !modelError ? <span><Check size={13} />{modelFeedback}</span> : null}</div>
          {modelError && <p className="model-error" role="alert">{modelError}</p>}
          {model?.training.error && <p className="model-error" role="alert">{model.training.error}</p>}
          {(training || !!model?.training.logs.length) && <details className="training-details"><summary>Training activity<ChevronDown size={13} /></summary><pre>{model?.training.logs.join('\n') || model?.training.phase || 'Preparing training…'}</pre></details>}
        </div>
        <div className="local-note"><ShieldCheck size={15} /><span>Runs on your machine</span></div>
        <span className="sidebar-version">AI Content Breaker / M1–M5</span>
      </div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><span className="breadcrumb">Workbench<span>/</span><strong>Text analysis</strong></span><span className="local-badge"><span className="status-dot ready" />Local workspace</span></header>
      <main id="workspace">
        <div className="page-heading"><div><h1>Text analysis</h1><p>Analyze text and inspect the output of each pipeline stage.</p></div><span className="project-label">Compiler design<br /><strong>M1–M5 pipeline</strong></span></div>
        <div className={`pipeline-rail ${analyzing ? 'pipeline-running' : ''}`} aria-label="Five-stage analysis pipeline">
          {stages.map(({ id, name, description, icon: Icon }) => <a key={id} href={result ? `#stage-${id}` : '#pipeline-guide'} onClick={() => { if (result) setView('pipeline'); else if (guide.current) guide.current.open = true }}><span className="rail-icon"><Icon size={18} strokeWidth={1.7} /></span><span className="rail-text"><span><small>{id}</small>{name}</span><span>{description}</span></span><ArrowRight size={13} className="rail-arrow" /></a>)}
        </div>

        <div className="workspace-grid">
          <section className="source-panel" aria-labelledby="source-heading">
            <div className="panel-heading"><div><FileText size={18} /><h2 id="source-heading">Source text</h2></div><span className="small-label">English</span></div>
            <form onSubmit={analyze} noValidate>
              <div className="editor-toolbar"><label htmlFor="source-text">Your passage</label><button type="button" className="text-button" onClick={loadSample} disabled={sampleLoading || analyzing}>{sampleLoading ? <LoaderCircle size={13} className="spin" /> : <Clipboard size={13} />}Load sample</button></div>
              <div className="textarea-wrap"><textarea ref={input} id="source-text" placeholder="Paste or write your text here…" value={text} onChange={event => { setText(event.target.value); if (error) setError('') }} spellCheck={false} aria-describedby={`input-help${error ? ' input-error' : ''}`} aria-invalid={!!error} disabled={analyzing || sampleLoading} maxLength={maxCharacters} /><span className="editor-corner" aria-hidden="true"><Braces size={24} strokeWidth={1} /></span></div>
              <div className="editor-meta"><span><strong>{wordCount.toLocaleString('en-IN')}</strong> words<span className="meta-divider">/</span>{text.length.toLocaleString('en-IN')} characters</span><button type="button" className="clear-button" disabled={!text || analyzing || sampleLoading} onClick={() => { setText(''); setError(''); input.current?.focus() }} aria-label="Clear source text"><Trash2 size={14} />Clear</button></div>
              <div className={`input-help ${wordCount >= minWords ? 'input-ready' : ''}`} id="input-help">{wordCount >= minWords ? <CheckCheck size={15} /> : <CircleHelp size={15} />}<span>{wordCount >= minWords ? 'Enough text for a verdict.' : `At least ${minWords} words are needed for a verdict.`}</span></div>
              <div className="form-feedback">{error && <p id="input-error" className="inline-error" role="alert"><TriangleAlert size={16} />{error}</p>}</div>
              <button type="submit" className="button button-primary analyze-button" disabled={analyzing || sampleLoading}><span>{analyzing ? <LoaderCircle size={17} className="spin" /> : <Play size={16} fill="currentColor" />}{analyzing ? 'Analyzing text…' : 'Analyze text'}</span><ArrowRight size={17} /></button>
            </form>
            <div className="source-footnote"><ShieldCheck size={14} /><p>Analysis runs locally. Your passage isn’t saved.</p></div>
          </section>

          <section className={`results-panel ${result ? 'has-result' : ''}`} aria-labelledby="results-heading" aria-busy={analyzing}>
            <div className="panel-heading"><div><Microscope size={18} /><h2 id="results-heading" ref={resultHeading} tabIndex={-1}>Analysis report</h2></div><span className={`report-badge ${result ? 'complete' : ''}`}>{analyzing ? 'Processing' : result ? stale ? 'Text edited' : 'Complete' : 'Awaiting text'}</span></div>
            {!result ? <div className="empty-report"><div className={`empty-mark ${analyzing ? 'empty-mark-loading' : ''}`}><ScanText size={34} strokeWidth={1.25} /><span /><span /></div><h3>{analyzing ? 'Analyzing text…' : 'No analysis yet'}</h3><p>{analyzing ? 'Running the lexer, parser, feature extractor, and scorers.' : 'Enter text and select Analyze text to view the results.'}</p></div> : <>
              {stale && <p className="stale-notice"><TriangleAlert size={15} />This report is for the previous text. Analyze again to update it.</p>}
              <div className="report-overview"><div className="overview-heading"><h3>{result.summary.abstain ? 'No verdict: fewer than 50 words' : result.summary.model_verdict === 'ai' ? 'Model verdict: AI' : result.summary.model_verdict === 'human' ? 'Model verdict: Human' : 'Model verdict unavailable'}</h3><span><Check size={12} />{format(result.summary.elapsed_ms / 1000, 2)}s</span></div>{result.summary.abstain && <p className="abstain-note">This passage has {result.summary.n_words} words. Both scorers abstain below {result.summary.min_words} words.</p>}<Score label="M5b · Trained model" value={result.summary.model_score} verdict={result.summary.model_verdict} abstain={result.summary.abstain} /><Score label="M5a · Rule-based scorer" value={result.summary.rule_score} verdict={result.summary.rule_verdict} abstain={result.summary.abstain} /><p className="score-note">Scores are model outputs, not proof of authorship.</p><dl className="metric-grid report-metrics"><Metric label="Words" value={result.summary.n_words} /><Metric label="Sentences" value={result.summary.n_sentences} /><Metric label="Unique words" value={result.summary.n_unique_words} /></dl></div>
              <div className="report-view-bar"><div className="view-switch" aria-label="Report view"><button aria-pressed={view === 'pipeline'} onClick={() => setView('pipeline')}><GitBranch size={14} />Pipeline</button><button aria-pressed={view === 'console'} onClick={() => setView('console')}><Terminal size={14} />Console</button></div><button className="copy-button" onClick={copyReport} aria-label={copied ? 'Report copied' : 'Copy terminal report'}>{copied ? <Check size={14} /> : <Copy size={14} />}<span role="status">{copied ? 'Copied' : 'Copy report'}</span></button></div>
              {view === 'pipeline' ? <PipelineReport result={result} /> : <div className="console-report"><div><Terminal size={13} />Original pipeline output</div><pre tabIndex={0}>{result.console_output}</pre></div>}
            </>}
          </section>
        </div>

        <details className="pipeline-guide" ref={guide} id="pipeline-guide"><summary><span><BookOpen size={17} /><strong>Pipeline guide</strong><span>M1–M5 stages</span></span><span className="guide-toggle">View stages<ArrowDown size={15} /></span></summary><div className="guide-content">{stages.map(({ id, name, icon: Icon }, i) => <div key={id}><span><Icon size={19} /><small>{id}</small></span><h3>{name}</h3><p>{['A maximal-munch lexer identifies words, numbers, punctuation, and sentence endings.', 'A symbol table tracks each word’s frequency and position in the token stream.', 'A recursive-descent grammar builds sentence trees. POS automata detect structural templates.', 'The feature extractor turns lexical and syntactic attributes into 89 measurable signals.', 'Nine inspectable rules and a trained classifier independently score the same passage.'][i]}</p></div>)}</div></details>
        <footer className="page-footer"><span>AI Content Breaker</span><span>Experimental stylometric analysis</span></footer>
      </main>
    </div>
  </div>
}
