import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [question, setQuestion] = useState("What do you see in this image?");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("analyze");
  const [error, setError] = useState(null);
  const [cfResult, setCfResult] = useState(null);
  const [maskX1, setMaskX1] = useState(0.25);
  const [maskY1, setMaskY1] = useState(0.25);
  const [maskX2, setMaskX2] = useState(0.75);
  const [maskY2, setMaskY2] = useState(0.75);
  const [pvResult, setPvResult] = useState(null);
  const [questions, setQuestions] = useState("What do you see?|Describe this image.|What is in the image?");

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setCfResult(null);
      setPvResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return alert("Please upload an image first!");
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", question);
      const res = await axios.post(API_URL + "/analyze", formData);
      setResult(res.data);
    } catch (e) {
      setError("Analysis failed. Make sure the API is running.");
    }
    setLoading(false);
  };

  const handleCounterfactual = async () => {
    if (!file) return alert("Please upload an image first!");
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", question);
      formData.append("x1", maskX1);
      formData.append("y1", maskY1);
      formData.append("x2", maskX2);
      formData.append("y2", maskY2);
      formData.append("mask_type", "black");
      const res = await axios.post(API_URL + "/counterfactual", formData);
      setCfResult(res.data);
    } catch (e) {
      setError("Counterfactual failed.");
    }
    setLoading(false);
  };

  const handlePromptVariations = async () => {
    if (!file) return alert("Please upload an image first!");
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("questions", questions);
      const res = await axios.post(API_URL + "/prompt-variations", formData);
      setPvResult(res.data);
    } catch (e) {
      setError("Prompt variations failed.");
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>VeriTrace AI</h1>
        <p>Multimodal Reasoning Auditor</p>
      </header>

      <div className="upload-section">
        <div className="upload-box">
          <input type="file" accept="image/*" onChange={handleFileChange} id="file-input" style={{ display: "none" }} />
          <label htmlFor="file-input" className="upload-label">
            {preview ? (
              <img src={preview} alt="preview" className="preview-img" />
            ) : (
              <div className="upload-placeholder">
                <p>Click to upload image</p>
              </div>
            )}
          </label>
        </div>
        <div className="question-box">
          <label>Question:</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="question-input"
            placeholder="Ask something about the image..."
          />
        </div>
      </div>

      <div className="tabs">
        <button className={activeTab === "analyze" ? "tab active" : "tab"} onClick={() => setActiveTab("analyze")}>Analyze</button>
        <button className={activeTab === "counterfactual" ? "tab active" : "tab"} onClick={() => setActiveTab("counterfactual")}>Counterfactual</button>
        <button className={activeTab === "variations" ? "tab active" : "tab"} onClick={() => setActiveTab("variations")}>Prompt Variations</button>
      </div>

      {error && <div className="error">{error}</div>}
      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Model thinking... this may take 30-60 seconds</p>
        </div>
      )}

      {activeTab === "analyze" && !loading && (
        <div className="tab-content">
          <button className="run-btn" onClick={handleAnalyze}>Run Analysis</button>
          {result && result.success && (
            <div className="results">
              <div className="answer-box">
                <h3>Model Answer:</h3>
                <p>{result.answer}</p>
              </div>
              <div className="images-grid">
                <div className="img-card">
                  <h4>Original Image</h4>
                  <img src={"data:image/png;base64," + result.original_image} alt="original" />
                </div>
                <div className="img-card">
                  <h4>Attention Heatmap</h4>
                  <img src={"data:image/png;base64," + result.heatmap_image} alt="heatmap" />
                </div>
                <div className="img-card">
                  <h4>Attention Overlay</h4>
                  <img src={"data:image/png;base64," + result.overlay_image} alt="overlay" />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "counterfactual" && !loading && (
        <div className="tab-content">
          <div className="mask-controls">
            <h3>Mask Region</h3>
            <div className="sliders">
              <label>X1: {maskX1} <input type="range" min="0" max="1" step="0.05" value={maskX1} onChange={e => setMaskX1(parseFloat(e.target.value))} /></label>
              <label>Y1: {maskY1} <input type="range" min="0" max="1" step="0.05" value={maskY1} onChange={e => setMaskY1(parseFloat(e.target.value))} /></label>
              <label>X2: {maskX2} <input type="range" min="0" max="1" step="0.05" value={maskX2} onChange={e => setMaskX2(parseFloat(e.target.value))} /></label>
              <label>Y2: {maskY2} <input type="range" min="0" max="1" step="0.05" value={maskY2} onChange={e => setMaskY2(parseFloat(e.target.value))} /></label>
            </div>
          </div>
          <button className="run-btn" onClick={handleCounterfactual}>Run Counterfactual</button>
          {cfResult && cfResult.success && (
            <div className="results">
              <div className="answer-box">
                <h3>Answer Changed: {cfResult.answer_changed ? "YES" : "NO"}</h3>
                <p><strong>Original:</strong> {cfResult.original_answer}</p>
                <p><strong>After Masking:</strong> {cfResult.masked_answer}</p>
              </div>
              <div className="images-grid">
                <div className="img-card">
                  <h4>Original</h4>
                  <img src={"data:image/png;base64," + cfResult.original_image} alt="original" />
                </div>
                <div className="img-card">
                  <h4>Masked</h4>
                  <img src={"data:image/png;base64," + cfResult.masked_image} alt="masked" />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "variations" && !loading && (
        <div className="tab-content">
          <div className="pv-controls">
            <label>Questions (separated by |):</label>
            <textarea
              value={questions}
              onChange={(e) => setQuestions(e.target.value)}
              className="questions-input"
              rows={3}
            />
          </div>
          <button className="run-btn" onClick={handlePromptVariations}>Run Variations</button>
          {pvResult && pvResult.success && (
            <div className="results">
              <div className="variations-list">
                {pvResult.results.map((r, i) => (
                  <div key={i} className="variation-item">
                    <p className="q-text">Q: {r.question}</p>
                    <p className="a-text">A: {r.answer}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;