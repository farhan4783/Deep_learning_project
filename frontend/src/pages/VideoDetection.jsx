import { useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, AlertCircle } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import ResultsDisplay from '../components/ResultsDisplay'
import { detectVideo } from '../services/api'
import '../pages/ImageDetection.css'

const VideoDetection = () => {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [settings, setSettings] = useState({
        frameSampling: 'uniform',
        numFrames: 32
    })

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile)
        setResult(null)
        setError(null)
    }

    const handleAnalyze = async () => {
        if (!file) return

        setLoading(true)
        setError(null)

        try {
            const data = await detectVideo(
                file,
                settings.frameSampling,
                settings.numFrames,
                false
            )
            setResult(data)
        } catch (err) {
            setError(err.message || 'Failed to analyze video')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="detection-page">
            <div className="container">
                <motion.div
                    className="page-header"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1>Video Deepfake Detection</h1>
                    <p>Upload a video to analyze it for signs of manipulation</p>
                </motion.div>

                <div className="detection-grid">
                    <motion.div
                        className="upload-section glass-card"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <h3>Upload Video</h3>
                        <FileUpload
                            accept="video/*"
                            onFileSelect={handleFileSelect}
                            file={file}
                        />

                        {file && (
                            <>
                                <div className="settings-section">
                                    <h4>Analysis Settings</h4>

                                    <div className="setting-group">
                                        <label>Frame Sampling</label>
                                        <select
                                            className="input-field"
                                            value={settings.frameSampling}
                                            onChange={(e) => setSettings({ ...settings, frameSampling: e.target.value })}
                                        >
                                            <option value="uniform">Uniform</option>
                                            <option value="random">Random</option>
                                            <option value="keyframes">Keyframes</option>
                                        </select>
                                    </div>

                                    <div className="setting-group">
                                        <label>Number of Frames</label>
                                        <input
                                            type="number"
                                            className="input-field"
                                            value={settings.numFrames}
                                            onChange={(e) => setSettings({ ...settings, numFrames: parseInt(e.target.value) })}
                                            min="8"
                                            max="100"
                                        />
                                    </div>
                                </div>

                                <button
                                    className="btn btn-primary"
                                    onClick={handleAnalyze}
                                    disabled={loading}
                                    style={{ width: '100%', marginTop: '1rem' }}
                                >
                                    {loading ? (
                                        <>
                                            <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                                            Analyzing...
                                        </>
                                    ) : (
                                        'Analyze Video'
                                    )}
                                </button>
                            </>
                        )}

                        {error && (
                            <div className="alert alert-error">
                                <AlertCircle size={20} />
                                <span>{error}</span>
                            </div>
                        )}
                    </motion.div>

                    <motion.div
                        className="results-section"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        {result ? (
                            <ResultsDisplay result={result} type="video" />
                        ) : (
                            <div className="results-placeholder glass-card">
                                <Upload size={64} className="placeholder-icon" />
                                <h3>No Results Yet</h3>
                                <p>Upload a video and click "Analyze" to see results</p>
                            </div>
                        )}
                    </motion.div>
                </div>
            </div>
        </div>
    )
}

export default VideoDetection
