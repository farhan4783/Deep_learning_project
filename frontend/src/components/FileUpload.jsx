import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import { Upload, File, X } from 'lucide-react'
import './FileUpload.css'

const FileUpload = ({ accept, onFileSelect, file, multiple = false }) => {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            if (multiple) {
                onFileSelect(acceptedFiles)
            } else {
                onFileSelect(acceptedFiles[0])
            }
        }
    }, [onFileSelect, multiple])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept,
        multiple
    })

    const handleRemove = (e) => {
        e.stopPropagation()
        onFileSelect(null)
    }

    return (
        <div className="file-upload">
            <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
            >
                <input {...getInputProps()} />

                {file ? (
                    <motion.div
                        className="file-preview"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        {file.type?.startsWith('image/') ? (
                            <img
                                src={URL.createObjectURL(file)}
                                alt="Preview"
                                className="preview-image"
                            />
                        ) : (
                            <div className="file-info">
                                <File size={48} />
                                <p className="file-name">{file.name}</p>
                                <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            </div>
                        )}

                        <button
                            className="remove-btn"
                            onClick={handleRemove}
                            type="button"
                        >
                            <X size={20} />
                        </button>
                    </motion.div>
                ) : (
                    <div className="dropzone-content">
                        <Upload size={48} className="upload-icon" />
                        <p className="dropzone-text">
                            {isDragActive
                                ? 'Drop the file here'
                                : 'Drag & drop a file here, or click to select'}
                        </p>
                        <p className="dropzone-hint">
                            {accept === 'image/*' ? 'Supports: JPG, PNG, BMP' : 'Supports: MP4, AVI, MOV, MKV'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}

export default FileUpload
