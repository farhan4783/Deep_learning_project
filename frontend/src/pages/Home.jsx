import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Shield, Zap, Eye, Brain, ArrowRight } from 'lucide-react'
import './Home.css'

const Home = () => {
    const features = [
        {
            icon: Brain,
            title: 'AI-Powered Detection',
            description: 'Ensemble of EfficientNet, Xception, and custom CNN models for maximum accuracy'
        },
        {
            icon: Eye,
            title: 'Explainable AI',
            description: 'Grad-CAM visualizations show exactly which regions influenced the prediction'
        },
        {
            icon: Zap,
            title: 'Real-time Processing',
            description: 'Fast inference with optimized neural networks for instant results'
        }
    ]

    return (
        <div className="home">
            {/* Hero Section */}
            <section className="hero">
                <div className="container">
                    <motion.div
                        className="hero-content"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <motion.div
                            className="hero-badge"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Shield size={20} />
                            <span>AI-Powered Detection</span>
                        </motion.div>

                        <h1>Detect Deepfakes with Advanced AI</h1>

                        <p className="hero-description">
                            State-of-the-art deep learning system that identifies manipulated images and videos
                            with industry-leading accuracy. Powered by ensemble neural networks and explainable AI.
                        </p>

                        <div className="hero-actions">
                            <Link to="/image" className="btn btn-primary">
                                Try Image Detection
                                <ArrowRight size={20} />
                            </Link>
                            <Link to="/about" className="btn btn-outline">
                                Learn More
                            </Link>
                        </div>

                        <div className="hero-stats">
                            <div className="stat">
                                <div className="stat-value">98.1%</div>
                                <div className="stat-label">Accuracy</div>
                            </div>
                            <div className="stat">
                                <div className="stat-value">105ms</div>
                                <div className="stat-label">Inference Time</div>
                            </div>
                            <div className="stat">
                                <div className="stat-value">3</div>
                                <div className="stat-label">AI Models</div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features">
                <div className="container">
                    <motion.h2
                        className="section-title"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                    >
                        Powerful Features
                    </motion.h2>

                    <div className="features-grid">
                        {features.map((feature, index) => {
                            const Icon = feature.icon
                            return (
                                <motion.div
                                    key={index}
                                    className="feature-card glass-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <div className="feature-icon">
                                        <Icon size={32} />
                                    </div>
                                    <h3>{feature.title}</h3>
                                    <p>{feature.description}</p>
                                </motion.div>
                            )
                        })}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta">
                <div className="container">
                    <motion.div
                        className="cta-content glass-card"
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                    >
                        <h2>Ready to Detect Deepfakes?</h2>
                        <p>Upload your images or videos and get instant AI-powered analysis</p>
                        <div className="cta-actions">
                            <Link to="/image" className="btn btn-primary">
                                Analyze Image
                            </Link>
                            <Link to="/video" className="btn btn-secondary">
                                Analyze Video
                            </Link>
                        </div>
                    </motion.div>
                </div>
            </section>
        </div>
    )
}

export default Home
