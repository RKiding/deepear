import { useState } from 'react'
import type { FormEvent } from 'react'
import { captureLiteEvent } from '../analytics'

type FeedbackType = 'bug' | 'idea' | 'question'

export function FeedbackButton() {
  const [open, setOpen] = useState(false)
  const [type, setType] = useState<FeedbackType>('idea')
  const [rating, setRating] = useState(4)
  const [message, setMessage] = useState('')
  const [contact, setContact] = useState('')

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = message.trim()
    if (!trimmed) return

    captureLiteEvent('feedback_submitted', {
      feedback_type: type,
      rating,
      message: trimmed,
      has_contact: Boolean(contact.trim()),
      contact: contact.trim() || undefined,
    })

    setOpen(false)
    setMessage('')
    setContact('')
    setType('idea')
    setRating(4)
  }

  return (
    <>
      <button
        type="button"
        className="lite-feedback-trigger"
        onClick={() => {
          captureLiteEvent('feedback_opened')
          setOpen(true)
        }}
      >
        反馈
      </button>

      {open && (
        <div className="lite-feedback-overlay" onClick={() => setOpen(false)}>
          <div className="lite-feedback-modal" onClick={(event) => event.stopPropagation()}>
            <div className="lite-feedback-title">提交反馈</div>
            <form className="lite-feedback-form" onSubmit={onSubmit}>
              <label>
                类型
                <select value={type} onChange={(e) => setType(e.target.value as FeedbackType)}>
                  <option value="idea">建议</option>
                  <option value="bug">问题</option>
                  <option value="question">疑问</option>
                </select>
              </label>

              <label>
                评分（1-5）
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={rating}
                  onChange={(e) => setRating(Number(e.target.value))}
                />
              </label>

              <label>
                反馈内容
                <textarea
                  required
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="你希望我们改进什么？"
                />
              </label>

              <label>
                联系方式（可选）
                <input
                  type="text"
                  value={contact}
                  onChange={(e) => setContact(e.target.value)}
                  placeholder="邮箱 / Telegram / 微信"
                />
              </label>

              <div className="lite-feedback-actions">
                <button type="button" onClick={() => setOpen(false)}>
                  取消
                </button>
                <button type="submit">提交</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  )
}
