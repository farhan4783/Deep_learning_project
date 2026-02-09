import { Routes, Route } from 'react-router-dom'
import { motion } from 'framer-motion'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import ImageDetection from './pages/ImageDetection'
import VideoDetection from './pages/VideoDetection'
import BatchDetection from './pages/BatchDetection'
import About from './pages/About'
import './App.css'

function App() {
    return (
        <div className="app">
            <Navbar />
            <motion.main
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
            >
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/image" element={<ImageDetection />} />
                    <Route path="/video" element={<VideoDetection />} />
                    <Route path="/batch" element={<BatchDetection />} />
                    <Route path="/about" element={<About />} />
                </Routes>
            </motion.main>
        </div>
    )
}

export default App
