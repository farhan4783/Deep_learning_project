import axios from 'axios'

const API_BASE_URL = '/api/v1'

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
})

/**
 * Detect deepfake in an image
 */
export const detectImage = async (file, explain = true) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('explain', explain)

    const response = await api.post('/detect/image', formData)
    return response.data
}

/**
 * Detect deepfake in a video
 */
export const detectVideo = async (file, frameSampling = 'uniform', numFrames = 32, explain = false) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('frame_sampling', frameSampling)
    formData.append('num_frames', numFrames)
    formData.append('explain', explain)

    const response = await api.post('/detect/video', formData)
    return response.data
}

/**
 * Batch detect deepfakes in multiple files
 */
export const detectBatch = async (files, explain = false) => {
    const formData = new FormData()
    files.forEach(file => {
        formData.append('files', file)
    })
    formData.append('explain', explain)

    const response = await api.post('/detect/batch', formData)
    return response.data
}

/**
 * Health check
 */
export const healthCheck = async () => {
    const response = await api.get('/health')
    return response.data
}

export default api
