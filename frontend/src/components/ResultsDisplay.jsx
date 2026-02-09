import { motion } from 'framer-motion'
import { AlertCircle, CheckCircle, Clock, Eye } from 'lucide-react'
import './ResultsDisplay.css'

const ResultsDisplay = ({ result, type }) => {
    const isFake = result.is_fake
    const confidence = (result.confidence * 100).toFixed(1)

    return (
        <motion.div
            className="results-display glass-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
        >
            {/* Main Result */}
            <div className={`result-header ${isFake ? 'fake' : 'real'}`}>
                <div className="result-icon">
                    {isFake ? <AlertCircle size={48} /> : <CheckCircle size={48} />}
                </div>
                <div className="result-text">
                    <h2>{isFake ? 'Deepfake Detected' : 'Appears Authentic'}</h2>
                    <p className="confidence">
                        Confidence: <strong>{confidence}%</strong>
                    </p>
                </div>
            </div>

            {/* Prediction Scores */}
            <div className="prediction-scores">
                <h3>Prediction Breakdown</h3>
                <div className="score-bars">
                    <div className="score-item">
                        <div className="score-label">
                            <span>Real</span>
                            <span>{(result.prediction_scores.real * 100).toFixed(1)}%</span>
                        </div>
                        <div className="score-bar">
                            <motion.div
                                className="score-fill real"
                                initial={{ width: 0 }}
                                animate={{ width: `${result.prediction_scores.real * 100}%` }}
                                transition={{ duration: 0.8, delay: 0.2 }}
                            />
                        </div>
                    </div>

                    <div className="score-item">
                        <div className="score-label">
                            <span>Fake</span>
                            <span>{(result.prediction_scores.fake * 100).toFixed(1)}%</span>
                        </div>
                        <div className="score-bar">
                            <motion.div
                                className="score-fill fake"
                                initial={{ width: 0 }}
                                animate={{ width: `${result.prediction_scores.fake * 100}%` }}
                                transition={{ duration: 0.8, delay: 0.3 }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Model Predictions */}
            {result.model_predictions && (
                <div className="model-predictions">
                    <h3>Individual Model Predictions</h3>
                    <div className="models-grid">
                        {Object.entries(result.model_predictions).map(([model, scores]) => {
                            if (typeof scores === 'object' && scores.fake !== undefined) {
                                return (
                                    <div key={model} className="model-card">
                                        <h4>{model.replace('_', ' ').toUpperCase()}</h4>
                                        <div className="model-score">
                                            <span className="score-value">{(scores.fake * 100).toFixed(1)}%</span>
                                            <span className="score-text">Fake Probability</span>
                                        </div>
                                    </div>
                                )
                            }
                            return null
                        })}
                    </div>
                </div>
            )}

            {/* Heatmap */}
            {result.heatmap_base64 && (
                <div className="heatmap-section">
                    <h3>
                        <Eye size={20} />
                        Explainability Heatmap
                    </h3>
                    <p className="heatmap-description">
                        Highlighted regions show areas that influenced the AI's decision
                    </p>
                    <img
                        src={`data:image/png;base64,${result.heatmap_base64}`}
                        alt="Grad-CAM Heatmap"
                        className="heatmap-image"
                    />
                </div>
            )}

            {/* Processing Time */}
            <div className="processing-info">
                <Clock size={16} />
                <span>Processed in {(result.processing_time * 1000).toFixed(0)}ms</span>
            </div>
        </motion.div>
    )
}

export default ResultsDisplay
