import { Zap, Search, BarChart2, FileText, CheckCircle2 } from 'lucide-react'
import './PhaseIndicator.css'

interface Props {
    phase: string
    progress: number
    status: string
}

const PHASES = [
    { id: 1, label: '初始化', icon: Zap, match: ['初始化'] },
    { id: 2, label: '热点扫描', icon: Search, match: ['热点扫描', '多源抓取', '意图分析'] },
    { id: 3, label: '深度分析', icon: BarChart2, match: ['金融分析', '分析信号'] },
    { id: 4, label: '生成报告', icon: FileText, match: ['生成报告', '完成'] }
]

export function PhaseIndicator({ phase, progress, status }: Props) {
    // Determine current active step index
    const currentStepIndex = PHASES.findIndex(p => p.match.some(m => phase.includes(m)))

    // Fallback if phase name doesn't match exactly
    const activeIndex = currentStepIndex !== -1 ? currentStepIndex :
        phase.includes('完成') ? 3 : 0

    return (
        <div className="phase-indicator">
            <div className="phase-steps">
                {PHASES.map((step, index) => {
                    const isActive = index === activeIndex
                    const isCompleted = index < activeIndex || status === 'completed'
                    const Icon = step.icon

                    return (
                        <div
                            key={step.id}
                            className={`phase-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                        >
                            <div className="step-icon">
                                {isCompleted ? <CheckCircle2 size={16} /> : <Icon size={16} />}
                            </div>
                            <span className="step-label">{step.label}</span>
                            {index < PHASES.length - 1 && (
                                <div className={`step-line ${index < activeIndex ? 'filled' : ''}`} />
                            )}
                        </div>
                    )
                })}
            </div>

            {status === 'running' && (
                <div className="phase-detail">
                    <span className="phase-name">{phase}</span>
                    <div className="progress-bar">
                        <div
                            className="progress-fill"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                    <span className="progress-text">{progress}%</span>
                </div>
            )}
        </div>
    )
}
