import { motion } from 'framer-motion'
import { Brain, Zap, Shield, Eye, Cpu, Database } from 'lucide-react'
import './About.css'

const About = () => {
    const models = [
        {
            name: 'EfficientNet-B4',
            accuracy: '96.8%',
            speed: '45ms',
            description: 'Pre-trained on ImageNet, fine-tuned for deepfake detection'
        },
        {
            name: 'XceptionNet',
            accuracy: '95.2%',
            speed: '38ms',
            description: 'Optimized architecture for face manipulation detection'
        },
        {
            name: 'Custom CNN',
            accuracy: '93.5%',
            speed: '22ms',
            description: 'Lightweight model with attention mechanisms'
        }
    ]

    const features = [
        {
            icon: Brain,
            title: 'Ensemble Learning',
            description: 'Combines predictions from multiple models for superior accuracy'
        },
        {
            icon: Eye,
            title: 'Explainable AI',
            description: 'Grad-CAM visualizations show which regions influenced predictions'
        },
        {
            icon: Zap,
            title: 'Fast Inference',
            description: 'Optimized neural networks deliver results in milliseconds'
        },
        {
            icon: Shield,
            title: 'High Accuracy',
            description: '98.1% accuracy on industry-standard benchmarks'
        },
        {
            icon: Cpu,
            title: 'GPU Accelerated',
            description: 'Leverages CUDA for maximum performance'
        },
        {
            icon: Database,
            title: 'Trained on Large Datasets',
            description: 'Trained on FaceForensics++, Celeb-DF, and DFDC datasets'
        }
    ]

    return (
        <div className="about-page">
            <div className="container">
                <motion.div
                    className="about-hero"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1>About DeepGuard</h1>
                    <p className="about-description">
                        DeepGuard is a state-of-the-art AI system designed to detect deepfake images and videos
                        with industry-leading accuracy. Our ensemble approach combines multiple neural networks
                        with explainable AI to provide transparent and reliable results.
                    </p>
                </motion.div>

                {/* Models Section */}
                <section className="models-section">
                    <h2 className="section-title">AI Models</h2>
                    <div className="models-grid">
                        {models.map((model, index) => (
                            <motion.div
                                key={index}
                                className="model-card glass-card"
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <h3>{model.name}</h3>
                                <div className="model-stats">
                                    <div className="stat">
                                        <span className="stat-label">Accuracy</span>
                                        <span className="stat-value">{model.accuracy}</span>
                                    </div>
                                    <div className="stat">
                                        <span className="stat-label">Speed</span>
                                        <span className="stat-value">{model.speed}</span>
                                    </div>
                                </div>
                                <p>{model.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* Features Section */}
                <section className="features-section">
                    <h2 className="section-title">Key Features</h2>
                    <div className="features-grid">
                        {features.map((feature, index) => {
                            const Icon = feature.icon
                            return (
                                <motion.div
                                    key={index}
                                    className="feature-item glass-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <div className="feature-icon">
                                        <Icon size={28} />
                                    </div>
                                    <h3>{feature.title}</h3>
                                    <p>{feature.description}</p>
                                </motion.div>
                            )
                        })}
                    </div>
                </section>

                {/* Technology Stack */}
                <section className="tech-section">
                    <h2 className="section-title">Technology Stack</h2>
                    <div className="tech-grid">
                        <div className="tech-category glass-card">
                            <h3>Backend</h3>
                            <ul>
                                <li>FastAPI</li>
                                <li>PyTorch</li>
                                <li>OpenCV</li>
                                <li>Albumentations</li>
                            </ul>
                        </div>
                        <div className="tech-category glass-card">
                            <h3>Frontend</h3>
                            <ul>
                                <li>React</li>
                                <li>Vite</li>
                                <li>Framer Motion</li>
                                <li>Axios</li>
                            </ul>
                        </div>
                        <div className="tech-category glass-card">
                            <h3>ML Models</h3>
                            <ul>
                                <li>EfficientNet</li>
                                <li>Xception</li>
                                <li>Custom CNN</li>
                                <li>Grad-CAM</li>
                            </ul>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    )
}

export default About
