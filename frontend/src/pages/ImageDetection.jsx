import { useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, AlertCircle, CheckCircle, XCircle } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import ResultsDisplay from '../components/ResultsDisplay'
import { detectImage } from '../services/api'
import './ImageDetection.css'

const ImageDetection = () => {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

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
            const data = await detectImage(file, true)
            setResult(data)
        } catch (err) {
            setError(err.message || 'Failed to analyze image')
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
                    <h1>Image Deepfake Detection</h1>
                    <p>Upload an image to analyze it for signs of manipulation</p>
                </motion.div>

                <div className="detection-grid">
                    {/* Upload Section */}
                    <motion.div
                        className="upload-section glass-card"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <h3>Upload Image</h3>
                        <FileUpload
                            accept="image/*"
                            onFileSelect={handleFileSelect}
                            file={file}
                        />

                        {file && (
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
                                    'Analyze Image'
                                )}
                            </button>
                        )}

                        {error && (
                            <div className="alert alert-error">
                                <AlertCircle size={20} />
                                <span>{error}</span>
                            </div>
                        )}
                    </motion.div>

                    {/* Results Section */}
                    <motion.div
                        className="results-section"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        {result ? (
                            <ResultsDisplay result={result} type="image" />
                        ) : (
                            <div className="results-placeholder glass-card">
                                <Upload size={64} className="placeholder-icon" />
                                <h3>No Results Yet</h3>
                                <p>Upload an image and click "Analyze" to see results</p>
                            </div>
                        )}
                    </motion.div>
                </div>
            </div>
        </div>
    )
}

export default ImageDetection
