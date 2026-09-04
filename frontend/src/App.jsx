import React, { useEffect, useState } from "react";
import "./App.css";

const API_URL = "https://idshield-backend-docker.onrender.com";

const navItems = [
  { id: "screening", label: "Screening", icon: "⌂" },
  { id: "history", label: "Case history", icon: "▣" },
  { id: "evidence", label: "Evidence", icon: "◇" },
  { id: "audit", label: "Audit trail", icon: "◉" },
];

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error) {
    console.error("IDShield UI error:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          background: "#0b141e",
          color: "#e8f1f6",
          padding: "32px",
          fontFamily: "system-ui, sans-serif"
        }}>
          <div style={{
            maxWidth: "700px",
            padding: "28px",
            border: "1px solid #334b5b",
            borderRadius: "14px",
            background: "#142534"
          }}>
            <h2 style={{ marginTop: 0 }}>IDShield UI error</h2>
            <p style={{ color: "#aebfc9", lineHeight: 1.6 }}>
              The backend may have responded, but the dashboard hit a rendering
              error. Open the browser console for the exact error.
            </p>
            <pre style={{
              whiteSpace: "pre-wrap",
              color: "#f0a0a0",
              fontSize: "12px"
            }}>{String(this.state.error?.message || this.state.error)}</pre>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [activeNav, setActiveNav] = useState("screening");
  const [activeTab, setActiveTab] = useState("overview");
  const [dragging, setDragging] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("idshield-theme") || "light";
    } catch {
      return "light";
    }
  });
  const [density, setDensity] = useState(() => {
    try {
      return localStorage.getItem("idshield-density") || "comfortable";
    } catch {
      return "comfortable";
    }
  });
  const [activeStep, setActiveStep] = useState(1);
  const [caseHistory, setCaseHistory] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("idshield-case-history") || "[]");
      return Array.isArray(saved) ? saved : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("idshield-theme", theme);
      localStorage.setItem("idshield-density", density);
      localStorage.setItem("idshield-case-history", JSON.stringify(caseHistory.slice(0, 12)));
    } catch {}
  }, [theme, density, caseHistory]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.density = density;
  }, [theme, density]);

  const analysis = result?.analysis || {};
  const risk = analysis.risk || {};
  const documentData = analysis.document || {};
  const tampering = analysis.tampering || {};
  const face = analysis.face_verification || {};
  const ocr = analysis.ocr || {};
  const mrz = analysis.mrz || {};
  const fields = analysis.visible_fields || {};
  const consistency = analysis.consistency || [];
  const mismatches = consistency.filter((x) => x.status === "mismatch");

  const riskLevel = risk.level || "unknown";
  const riskLabel =
    riskLevel === "low"
      ? "LOW RISK"
      : riskLevel === "medium"
        ? "MEDIUM RISK"
        : riskLevel === "high"
          ? "HIGH RISK"
          : riskLevel === "unsupported"
            ? "UNSUPPORTED DOCUMENT"
            : "AWAITING";

  const consistencyLabel =
    mismatches.length > 0
      ? "Mismatch detected"
      : consistency.some((x) => x.status === "match")
        ? "MRZ / fields consistent"
        : "Review required";

  const tamperStatus = tampering.status || "not_checked";
  const tamperLabel =
    tamperStatus === "low"
      ? "No obvious indicators"
      : tamperStatus === "medium"
        ? "Review indicators"
        : tamperStatus === "high"
          ? "Multiple indicators"
          : "Not checked";

  const faceDetected = face.status === "detected";
  const mrzDetected = Boolean(mrz.detected);

  const displayFields = {
    surname: mrz?.parsed?.surname || fields.surname,
    given_names: mrz?.parsed?.given_names || fields.given_names,
    nationality: mrz?.parsed?.nationality || fields.nationality,
    date_of_birth: mrz?.parsed?.date_of_birth || fields.date_of_birth,
    date_of_expiry: mrz?.parsed?.date_of_expiry || fields.date_of_expiry,
    passport_number: mrz?.parsed?.passport_number || fields.passport_number,
  };

  function acceptFile(selected) {
    if (!selected) return;
    setFile(selected);
    setResult(null);
    setError("");
    setActiveNav("screening");

    const url = URL.createObjectURL(selected);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
  }

  function handleFileChange(event) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  }

  async function handleAnalyze() {
    if (!file || isAnalyzing) return;

    setIsAnalyzing(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json()
        : null;

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            `Analysis failed (HTTP ${response.status}).`
        );
      }

      if (!data) {
        throw new Error(
          "The backend returned an unexpected non-JSON response."
        );
      }

      setResult(data);

      const resultAnalysis = data?.analysis || {};
      const resultDocument = resultAnalysis.document || {};
      const resultRisk = resultAnalysis.risk || {};
      const resultVerification = resultAnalysis.verification || {};
      const caseRecord = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        filename: file.name,
        documentType: resultDocument.type || "unknown",
        status: resultDocument.status || "screened",
        riskLevel: resultRisk.level || "unknown",
        riskScore: resultRisk.score ?? null,
        decision: resultVerification.decision || "review",
        timestamp: new Date().toISOString(),
      };
      setCaseHistory((items) => [caseRecord, ...items.filter((item) => item.filename !== file.name || item.timestamp !== caseRecord.timestamp)].slice(0, 12));

      setActiveNav("screening");
      setActiveTab("overview");
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to IDShield backend. Make sure FastAPI is running on port 8001."
      );
    } finally {
      setIsAnalyzing(false);
    }
  }

  function navigate(id) {
    setActiveNav(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className={`app-shell theme-${theme} density-${density}`}>
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-head">
          <div className="brand">
            <div className="brand-mark">
              <span>I</span>
              <span>D</span>
            </div>
            <div className="brand-copy">
              <h1>IDShield</h1>
              <p>OFFICER CONSOLE</p>
            </div>
          </div>

          <button
            type="button"
            className="sidebar-toggle"
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>

        <nav className="side-nav" aria-label="Primary navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${activeNav === item.id ? "active" : ""}`}
              onClick={() => navigate(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="secure-session">
            <span className="online-dot sidebar-status-dot" />
            <div>
              <strong>Secure session</strong>
              <small>AI assists · officer decides</small>
            </div>
          </div>
          <div className="sidebar-version">IDSHIELD · SIH 2026</div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="topbar-title">
            <span>SCREENING / {activeNav === "screening" ? "NEW CASE" : prettyNav(activeNav)}</span>
            <h2>
              {activeNav === "screening"
                ? "Identity document screening"
                : prettyNav(activeNav)}
            </h2>
          </div>

          <div className="topbar-right">
            <div className="system-online">
              <span className="online-dot" />
              SYSTEM ONLINE
            </div>

            <div className="view-controls">
              <button
                type="button"
                className={`density-switch ${density}`}
                onClick={() =>
                  setDensity((value) =>
                    value === "comfortable" ? "compact" : "comfortable"
                  )
                }
                title={`Switch to ${density === "comfortable" ? "compact" : "comfortable"} density`}
              >
                <span className="density-icon">▦</span>
                <span>{density === "comfortable" ? "Comfortable" : "Compact"}</span>
              </button>

              <button
                type="button"
                className={`theme-toggle ${theme}`}
                onClick={() =>
                  setTheme((value) => (value === "light" ? "dark" : "light"))
                }
                aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
              >
                <span className="theme-icon">{theme === "light" ? "☾" : "☀"}</span>
                <span>{theme === "light" ? "Dark" : "Light"}</span>
              </button>
            </div>

            <div className="user-chip">TO</div>
          </div>
        </header>

        {activeNav === "screening" ? (
          <ScreeningView
            file={file}
            preview={preview}
            result={result}
            isAnalyzing={isAnalyzing}
            error={error}
            dragging={dragging}
            setDragging={setDragging}
            handleFileChange={handleFileChange}
            handleDrop={handleDrop}
            clearFile={clearFile}
            handleAnalyze={handleAnalyze}
            risk={risk}
            riskLevel={riskLevel}
            riskLabel={riskLabel}
            documentData={documentData}
            ocr={ocr}
            mrz={mrz}
            tampering={tampering}
            face={face}
            faceDetected={faceDetected}
            mrzDetected={mrzDetected}
            consistency={consistency}
            mismatches={mismatches}
            consistencyLabel={consistencyLabel}
            tamperLabel={tamperLabel}
            displayFields={displayFields}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            activeStep={activeStep}
            setActiveStep={setActiveStep}
          />
        ) : (
          <InfoView type={activeNav} result={result} caseHistory={caseHistory} />
        )}

        <footer className="footer">
          <span>IDShield · SIH 2026 · Prototype screening workspace</span>
          <span>AI assists screening — authorized personnel make the final decision.</span>
        </footer>
      </div>
    </div>
  );
}

function ScreeningView({
  file,
  preview,
  result,
  isAnalyzing,
  error,
  dragging,
  setDragging,
  handleFileChange,
  handleDrop,
  clearFile,
  handleAnalyze,
  risk,
  riskLevel,
  riskLabel,
  documentData,
  ocr,
  mrz,
  tampering,
  face,
  faceDetected,
  mrzDetected,
  consistency,
  mismatches,
  consistencyLabel,
  tamperLabel,
  displayFields,
  activeTab,
  setActiveTab,
  activeStep,
  setActiveStep,
}) {
  const riskScore = Number(risk?.score ?? 0);
  const progress = isAnalyzing ? 70 : result ? 100 : 0;

  return (
    <main className="content">
      <section className="pipeline-card">
        {[
          ["01", "Upload", "Document image"],
          ["02", "Extract", "OCR + MRZ"],
          ["03", "Validate", "Rules + consistency"],
          ["04", "Analyze", "Forensics + face"],
          ["05", "Assess", "Risk + evidence"],
        ].map(([number, title, sub], index, arr) => {
          const step = index + 1;
          const enabled = step === 1 || Boolean(result);
          const isActive = step === activeStep;
          const isDone = result && step < 5;

          return (
            <div className="pipeline-step" key={number}>
              <button
                type="button"
                className={`step-button ${enabled ? "enabled" : "disabled"} ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}
                disabled={!enabled}
                onClick={() => enabled && setActiveStep(step)}
              >
                <span className="step-circle">{isDone ? "✓" : number}</span>
                <span>
                  <strong>{title}</strong>
                  <small>{sub}</small>
                </span>
              </button>
              {index < arr.length - 1 && <i />}
            </div>
          );
        })}
      </section>

      <section className={`intro ${file || result ? "case-active" : ""}`}>
        <div>
          <p className="eyebrow">AI-ASSISTED DOCUMENT SCREENING</p>
          <h1 className="hero-title">
            <span className="hero-title-line">From document image</span>
            <span className="hero-title-line accent">to explainable decision.</span>
          </h1>
          <p className="intro-copy">
            One workspace for extraction, validation, visual forensics,
            face verification and evidence-backed risk assessment.
          </p>
        </div>
        <div className="sih-badge" aria-label="Problem statement reference">
          <span className="sih-badge-kicker">SIH PROTOTYPE</span>
          <strong>SIH26188</strong>
          <span>Identity screening</span>
        </div>
      </section>

      <section className="workspace">
        <article className="panel upload-panel">
          <PanelHeader
            number="01"
            title="Document"
            badge="JPG · PNG · WEBP · BMP"
          />

          <label
            className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp"
              onChange={handleFileChange}
            />

            {preview ? (
              <>
                <img src={preview} alt="Uploaded identity document preview" />
                <div className="preview-overlay">
                  <strong>DOCUMENT PREVIEW</strong>
                  <span>Click to replace</span>
                </div>
              </>
            ) : (
              <div className="drop-empty">
                <div className="upload-glyph">↑</div>
                <strong>Upload identity document</strong>
                <span>Drag & drop or click to browse</span>
                <small>IMAGE FILES · MAX 10 MB</small>
              </div>
            )}
          </label>

          {file && (
            <div className="file-meta">
              <div className="file-icon">▧</div>
              <div>
                <strong>{file.name}</strong>
                <small>{formatBytes(file.size)} · Ready for screening</small>
              </div>
              <button type="button" onClick={clearFile} aria-label="Remove document">
                ×
              </button>
            </div>
          )}

          {error && <div className="error-box">{error}</div>}

          <button
            className={`primary-button${result?.analysis?.document?.status === "unsupported" ? " is-secondary" : ""}`}
            type="button"
            disabled={!file || isAnalyzing || result?.analysis?.document?.status === "unsupported"}
            title={result?.analysis?.document?.status === "unsupported" ? "Choose another document to continue" : undefined}
            onClick={handleAnalyze}
          >
            <span>
              {isAnalyzing
                ? "ANALYZING DOCUMENT..."
                : result?.analysis?.document?.status === "unsupported"
                  ? "TRY ANOTHER DOCUMENT"
                  : "ANALYZE DOCUMENT"}
            </span>
            {!isAnalyzing && result?.analysis?.document?.status !== "unsupported" && <b>→</b>}
          </button>

          {isAnalyzing && (
            <div className="progress">
              <i style={{ width: `${progress}%` }} />
            </div>
          )}
        </article>

        <article className="panel result-panel">
          <PanelHeader
            number="02"
            title="Screening result"
            badge={result ? "ANALYSIS COMPLETE" : "○ AWAITING DOCUMENT"}
            badgeClass={result ? "complete" : ""}
          />

          {!result ? (
            <div className="empty-result">
              <div className="empty-shield">◇</div>
              <h3>Ready for screening</h3>
              <p>
                Upload a document on the left to begin OCR extraction,
                validation and forensic checks.
              </p>
              <div className="empty-tags">
                <span>OCR</span>
                <span>MRZ</span>
                <span>FORENSICS</span>
                <span>RISK</span>
              </div>
            </div>
          ) : documentData?.status === "unsupported" ? (
            <div className="unsupported-result">
              <div className="unsupported-icon">!</div>
              <div>
                <span className="eyebrow">DOCUMENT CLASSIFICATION</span>
                <h3>Unsupported or unrecognized document</h3>
                <p>
                  IDShield could not confidently identify this upload as a supported
                  identity document, so the screening pipeline was stopped.
                </p>
                <div className="unsupported-meta" style={{ display: "grid", gap: "8px", marginTop: "14px", marginBottom: "16px" }}>
                  <div><b>Detected:</b> {prettyType(documentData?.type)}</div>
                  <div><b>Classification confidence:</b> {documentData?.confidence ?? 0}%</div>
                </div>
                <button type="button" className="secondary-action" onClick={clearFile}>Try another document</button>
              </div>
            </div>
          ) : (
            <>
              <div className={`risk-card ${riskTone(riskLevel)}`}>
                <div>
                  <span>RISK SCORE</span>
                  <strong>{riskScore}<small>/100</small></strong>
                </div>
                <div className="risk-middle">
                  <b>{riskLabel}</b>
                  <small>{risk?.summary || "Screening completed."}</small>
                </div>
                <div
                  className="risk-ring"
                  style={{ "--score": `${Math.min(riskScore, 100) * 3.6}deg` }}
                >
                  <span>{riskScore}</span>
                </div>
              </div>

              <div className="quick-grid">
                <Quick label="DOCUMENT" value={prettyType(documentData?.type)} tone="good" />
                <Quick
                  label="OCR"
                  value={`${ocr?.confidence ?? "N/A"}%`}
                  tone={Number(ocr?.confidence) >= 70 ? "good" : "warn"}
                />
                <Quick
                  label="MRZ"
                  value={mrzDetected ? "DETECTED" : "NOT DETECTED"}
                  tone={mrzDetected ? "good" : "warn"}
                />
                <Quick
                  label="FACE"
                  value={faceDetected ? "DETECTED" : "REVIEW"}
                  tone={faceDetected ? "good" : "warn"}
                />
              </div>

              <div className="tabs">
                {[
                  ["overview", "Overview"],
                  ["fields", "Fields"],
                  ["forensics", "Forensics"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={activeTab === id ? "selected" : ""}
                    onClick={() => setActiveTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {activeTab === "overview" && (
                <div className="checks">
                  <Check
                    title="OCR / MRZ consistency"
                    status={mismatches.length ? "danger" : "success"}
                    text={consistencyLabel}
                  />
                  <Check
                    title="MRZ validation"
                    status={
                      risk?.checks?.find((x) => x.check === "mrz_checksums")?.status === "failed"
                        ? "warning"
                        : "success"
                    }
                    text={
                      risk?.checks?.find((x) => x.check === "mrz_checksums")?.message ||
                      (mrzDetected ? "Machine-readable zone detected." : "MRZ not detected.")
                    }
                  />
                  <Check
                    title="Tampering screening"
                    status={
                      tampering?.status === "high"
                        ? "danger"
                        : tampering?.status === "medium" || tampering?.status === "not_checked"
                          ? "warning"
                          : "success"
                    }
                    text={tamperLabel}
                  />
                  <Check
                    title="Face presence"
                    status={faceDetected ? "success" : "warning"}
                    text={
                      faceDetected
                        ? `${face.faces_detected || 1} face region detected`
                        : "Face not clearly detected"
                    }
                  />
                  <Check
                    title="Required fields"
                    status={
                      risk?.checks?.find((x) => x.check === "required_fields")?.status === "passed"
                        ? "success"
                        : "warning"
                    }
                    text={
                      risk?.checks?.find((x) => x.check === "required_fields")?.message ||
                      "Primary fields checked"
                    }
                  />
                  <Check
                    title="Document classification"
                    status={documentData?.type === "unknown" ? "warning" : "success"}
                    text={`${prettyType(documentData?.type)} detected (${documentData?.confidence ?? 0}% confidence)`}
                  />
                </div>
              )}

              {activeTab === "fields" && (
                <div className="field-grid compact">
                  <Field label="Surname" value={displayFields.surname} />
                  <Field label="Given names" value={displayFields.given_names} />
                  <Field label="Nationality" value={displayFields.nationality} />
                  <Field label="Date of birth" value={displayFields.date_of_birth} />
                  <Field label="Date of expiry" value={displayFields.date_of_expiry} />
                  <Field label="Passport number" value={displayFields.passport_number} />
                </div>
              )}

              {activeTab === "forensics" && (
                <div className="forensics-list">
                  <Forensic label="Tampering score" value={`${tampering?.score ?? 0}/45`} />
                  <Forensic label="Face status" value={faceDetected ? "Detected" : "Not detected"} />
                  <Forensic label="MRZ status" value={mrzDetected ? "Detected" : "Not detected"} />
                  <Forensic
                    label="Image quality"
                    value={risk?.checks?.find((x) => x.check === "image_quality")?.status || "Checked"}
                  />
                </div>
              )}
            </>
          )}
        </article>
      </section>

      <section className="evidence-panel">
        <div className="evidence-header">
          <div>
            <p className="eyebrow">03 · DECISION SUPPORT</p>
            <h2>Explainable evidence</h2>
            <p>
              Every result is presented as a screening indicator so an officer
              can understand why a case needs attention.
            </p>
          </div>
          <span className="human-badge">HUMAN-IN-THE-LOOP</span>
        </div>

        <div className="evidence-grid">
          <EvidenceCard
            label="DOCUMENT"
            value={result ? prettyType(documentData?.type) : "Waiting for analysis"}
            tone={result ? "good" : "neutral"}
          />
          <EvidenceCard
            label="OCR"
            value={result ? `Confidence ${ocr?.confidence ?? "N/A"}%` : "Waiting for analysis"}
            tone={result && Number(ocr?.confidence) >= 70 ? "good" : result ? "warn" : "neutral"}
          />
          <EvidenceCard
            label="CONSISTENCY"
            value={result ? consistencyLabel : "Waiting for analysis"}
            tone={result && mismatches.length ? "danger" : result ? "good" : "neutral"}
          />
          <EvidenceCard
            label="RECOMMENDATION"
            value={
              result
                ? riskLevel === "high" || riskLevel === "medium" || risk?.score > 15
                  ? "Manual review"
                  : "Screened low risk"
                : "No recommendation yet"
            }
            tone={
              !result
                ? "neutral"
                : riskLevel === "high" || riskLevel === "medium" || risk?.score > 15
                  ? "warn"
                  : "good"
            }
          />
        </div>

        {result && risk?.reasons?.length > 0 && (
          <div className="indicator-box">
            <div className="indicator-title">SCREENING INDICATORS</div>
            {risk.reasons.map((reason, index) => (
              <div className="indicator" key={index}>
                <span>!</span>
                <p>{reason}</p>
              </div>
            ))}
          </div>
        )}

        {result && (
          <div className="decision-strip">
            <div>
              <span>RECOMMENDED ACTION</span>
              <strong>
                {riskLevel === "high" || riskLevel === "medium" || risk?.score > 15
                  ? "MANUAL REVIEW"
                  : "CONTINUE STANDARD VERIFICATION"}
              </strong>
            </div>
            <p>
              IDShield provides screening indicators. It does not establish
              legal authenticity by itself.
            </p>
          </div>
        )}
      </section>
    </main>
  );
}

function InfoView({ type, result, caseHistory }) {
  const analysis = result?.analysis || {};
  const documentData = analysis.document || {};
  const risk = analysis.risk || {};
  const ocr = analysis.ocr || {};
  const mrz = analysis.mrz || {};
  const tampering = analysis.tampering || {};
  const face = analysis.face_verification || {};
  const verification = analysis.verification || {};
  const orientation = analysis.orientation || {};
  const checks = Array.isArray(risk.checks) ? risk.checks : [];

  const formatTime = (value) => {
    try {
      return new Date(value).toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
      });
    } catch {
      return value || "—";
    }
  };

  if (type === "history") {
    return (
      <main className="info-page">
        <div className="info-hero">
          <p className="eyebrow">CASE MANAGEMENT</p>
          <h1>Case history</h1>
          <p>Recent local screening sessions are retained in this browser so the officer can quickly revisit what was screened.</p>
        </div>
        {caseHistory?.length ? (
          <div className="info-table-wrap">
            <div className="info-table-head">
              <span>RECENT CASES</span>
              <span>{caseHistory.length} stored</span>
            </div>
            <div className="info-table">
              {caseHistory.map((item) => (
                <div className="info-table-row" key={item.id}>
                  <div>
                    <strong>{item.filename}</strong>
                    <small>{prettyType(item.documentType)} · {formatTime(item.timestamp)}</small>
                  </div>
                  <div className={`info-status ${item.riskLevel === "high" ? "danger" : item.riskLevel === "medium" ? "warn" : item.status === "unsupported" ? "neutral" : "good"}`}>
                    {item.status === "unsupported" ? "UNSUPPORTED" : `${String(item.riskLevel || "review").toUpperCase()}`}
                  </div>
                  <div className="info-score">{item.riskScore == null ? "—" : `${item.riskScore}/100`}</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="info-empty">
            <strong>No screened cases yet</strong>
            <span>Run a document screening and the case summary will appear here.</span>
          </div>
        )}
      </main>
    );
  }

  if (type === "evidence") {
    const reasons = Array.isArray(risk.reasons) ? risk.reasons : [];
    return (
      <main className="info-page">
        <div className="info-hero">
          <p className="eyebrow">EVIDENCE WORKSPACE</p>
          <h1>Evidence</h1>
          <p>Evidence from the current screening, grouped into extraction, document validation and visual indicators.</p>
        </div>
        {!result ? (
          <div className="info-empty"><strong>No current case</strong><span>Run a screening to populate evidence.</span></div>
        ) : documentData.status === "unsupported" ? (
          <div className="info-empty"><strong>Analysis stopped</strong><span>This upload was not confidently identified as a supported identity document.</span></div>
        ) : (
          <div className="info-cards">
            <div className="info-card"><span>01</span><h3>Document</h3><p>{prettyType(documentData.type)} · {documentData.confidence ?? 0}% classification confidence</p></div>
            <div className="info-card"><span>02</span><h3>Extraction</h3><p>OCR {ocr.confidence ?? "—"}% · MRZ {mrz.detected ? "detected" : "not detected"}</p></div>
            <div className="info-card"><span>03</span><h3>Forensics</h3><p>Tampering {tampering.status || "not checked"} · Face {face.status || "not checked"}</p></div>
          </div>
        )}
        {result && reasons.length > 0 && (
          <div className="info-detail-panel">
            <span className="eyebrow">SCREENING INDICATORS</span>
            {reasons.map((reason, index) => <div className="info-detail-row" key={index}><b>!</b><span>{reason}</span></div>)}
          </div>
        )}
      </main>
    );
  }

  return (
    <main className="info-page">
      <div className="info-hero">
        <p className="eyebrow">SECURITY & ACCOUNTABILITY</p>
        <h1>Audit trail</h1>
        <p>Local session events for the current prototype screening workflow.</p>
      </div>
      {!result ? (
        <div className="info-empty"><strong>No audit event yet</strong><span>Run a screening to create a local audit snapshot.</span></div>
      ) : (
        <div className="info-cards">
          <div className="info-card"><span>01</span><h3>Session</h3><p>File: {result.filename || "Current upload"}<br />Completed: {formatTime(new Date().toISOString())}</p></div>
          <div className="info-card"><span>02</span><h3>Decision</h3><p>{String(verification.decision || "review").replaceAll("_", " ")} · {risk.score == null ? "No score" : `${risk.score}/100`}</p></div>
          <div className="info-card"><span>03</span><h3>Processing</h3><p>Orientation: {orientation.method || "standard"} · MRZ: {mrz.detected ? "detected" : "not detected"}</p></div>
        </div>
      )}
      {result && checks.length > 0 && (
        <div className="info-detail-panel">
          <span className="eyebrow">CHECK LOG</span>
          {checks.slice(0, 8).map((check, index) => (
            <div className="info-detail-row" key={`${check.check || "check"}-${index}`}><b>{check.status === "passed" ? "✓" : "!"}</b><span><strong>{String(check.check || "check").replaceAll("_", " ")}</strong> — {check.message || "Recorded"}</span></div>
          ))}
        </div>
      )}
    </main>
  );
}

function PanelHeader({ number, title, badge, badgeClass = "" }) {
  return (
    <div className="panel-header">
      <div>
        <span className="panel-number">{number}</span>
        <h2>{title}</h2>
      </div>
      <span className={`panel-badge ${badgeClass}`}>{badge}</span>
    </div>
  );
}

function Quick({ label, value, tone }) {
  return (
    <div className="quick-card">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}

function Check({ title, status, text }) {
  const icon = status === "success" ? "✓" : status === "danger" ? "×" : "!";
  return (
    <div className="check-row">
      <span className={`check-icon ${status}`}>{icon}</span>
      <div>
        <strong>{title}</strong>
        <small>{text}</small>
      </div>
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div className="field-card">
      <span>{label}</span>
      <strong>{value || "Not extracted"}</strong>
    </div>
  );
}

function Forensic({ label, value }) {
  return (
    <div className="forensic-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvidenceCard({ label, value, tone }) {
  return (
    <div className={`evidence-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function riskTone(level) {
  if (level === "high") return "danger";
  if (level === "medium") return "watch";
  if (level === "low") return "safe";
  return "idle";
}

function prettyType(type) {
  if (!type || type === "unknown") return "Unknown document";
  return type.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function prettyNav(id) {
  return id.replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  const mb = bytes / 1024 / 1024;
  return mb >= 1 ? `${mb.toFixed(2)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function AppWithErrorBoundary() {
  return (
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  );
}

export default AppWithErrorBoundary;



