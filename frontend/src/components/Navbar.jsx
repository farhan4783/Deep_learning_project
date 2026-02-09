import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Shield, Image, Video, Layers, Info } from 'lucide-react'
import './Navbar.css'

const Navbar = () => {
    const location = useLocation()

    const navItems = [
        { path: '/', label: 'Home', icon: Shield },
        { path: '/image', label: 'Image', icon: Image },
        { path: '/video', label: 'Video', icon: Video },
        { path: '/batch', label: 'Batch', icon: Layers },
        { path: '/about', label: 'About', icon: Info },
    ]

    return (
        <motion.nav
            className="navbar"
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <div className="navbar-container">
                <Link to="/" className="navbar-logo">
                    <Shield className="logo-icon" />
                    <span className="logo-text">DeepGuard</span>
                </Link>

                <ul className="navbar-menu">
                    {navItems.map((item) => {
                        const Icon = item.icon
                        const isActive = location.pathname === item.path

                        return (
                            <li key={item.path}>
                                <Link
                                    to={item.path}
                                    className={`nav-link ${isActive ? 'active' : ''}`}
                                >
                                    <Icon size={18} />
                                    <span>{item.label}</span>
                                    {isActive && (
                                        <motion.div
                                            className="active-indicator"
                                            layoutId="activeIndicator"
                                            transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                                        />
                                    )}
                                </Link>
                            </li>
                        )
                    })}
                </ul>
            </div>
        </motion.nav>
    )
}

export default Navbar
