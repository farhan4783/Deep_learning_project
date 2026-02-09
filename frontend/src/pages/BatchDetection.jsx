import { useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, AlertCircle, Trash2 } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { detectBatch } from '../services/api'
import '../pages/ImageDetection.css'
import './BatchDetection.css'

const BatchDetection = () => {
    const [files, setFiles] = useState([])
    const [loading, setLoading] = useState(false)
    const [results, setResults] = useState(null)
    const [error, setError] = useState(null)

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.bmp'],
            'video/*': ['.mp4', '.avi', '.mov', '.mkv']
        },
        multiple: true,
        maxFiles: 20,
        onDrop: (acceptedFiles) => {
            setFiles(prev => [...prev, ...acceptedFiles].slice(0, 20))
            setResults(null)
            setError(null)
        }
    })

    const handleRemoveFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index))
    }

    const handleAnalyze = async () => {
        if (files.length === 0) return

        setLoading(true)
        setError(null)

        try {
            const data = await detectBatch(files, false)
            setResults(data.results)
        } catch (err) {
            setError(err.message || 'Failed to analyze files')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="detection-page batch-detection">
            <div className="container">
                <motion.div
                    className="page-header"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1>Batch Deepfake Detection</h1>
                    <p>Upload multiple images or videos for batch analysis (max 20 files)</p>
                </motion.div>

                <motion.div
                    className="batch-upload glass-card"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <div
                        {...getRootProps()}
                        className={`batch-dropzone ${isDragActive ? 'active' : ''}`}
                    >
                        <input {...getInputProps()} />
                        <Upload size={48} className="upload-icon" />
                        <p className="dropzone-text">
                            {isDragActive
                                ? 'Drop the files here'
                                : 'Drag & drop files here, or click to select'}
                        </p>
                        <p className="dropzone-hint">
                            {files.length} / 20 files selected
                        </p>
                    </div>

                    {files.length > 0 && (
                        <div className="files-list">
                            <div className="files-header">
                                <h3>Selected Files</h3>
                                <button
                                    className="btn btn-outline"
                                    onClick={() => setFiles([])}
                                    style={{ padding: '0.5rem 1rem', fontSize: '0.875rem' }}
                                >
                                    Clear All
                                </button>
                            </div>

                            <div className="files-grid">
                                {files.map((file, index) => (
                                    <div key={index} className="file-item">
                                        <div className="file-info">
                                            <p className="file-name">{file.name}</p>
                                            <p className="file-size">
                                                {(file.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        </div>
                                        <button
                                            className="remove-file-btn"
                                            onClick={() => handleRemoveFile(index)}
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>
                                ))}
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
                                        Analyzing {files.length} files...
                                    </>
                                ) : (
                                    `Analyze ${files.length} Files`
                                )}
                            </button>
                        </div>
                    )}

                    {error && (
                        <div className="alert alert-error">
                            <AlertCircle size={20} />
                            <span>{error}</span>
                        </div>
                    )}
                </motion.div>

                {results && (
                    <motion.div
                        className="batch-results"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <h2>Batch Results</h2>
                        <div className="results-grid">
                            {results.map((item, index) => (
                                <div key={index} className="batch-result-card glass-card">
                                    <h4>{item.filename}</h4>
                                    {item.error ? (
                                        <div className="result-error">
                                            <AlertCircle size={20} />
                                            <span>{item.error}</span>
                                        </div>
                                    ) : (
                                        <div className={`result-badge ${item.result.is_fake ? 'fake' : 'real'}`}>
                                            {item.result.is_fake ? 'Deepfake' : 'Authentic'}
                                            <span className="confidence">
                                                {(item.result.confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    )
}

export default BatchDetection
