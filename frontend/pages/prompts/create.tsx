import { useState } from 'react'
import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { useCreatePrompt } from '@/lib/hooks'
import { toast } from '@/lib/toast'

export default function CreatePromptPage() {
  const router = useRouter()
  const createPrompt = useCreatePrompt()

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    prompt_text: '',
    prompt_type: 'test_prompt',
    target: '',
    category: '',
    tags: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const tags = formData.tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean)
      const finalTags = [
        ...tags,
        ...(formData.prompt_type === 'system_prompt' && formData.target === 'evaluator' ? ['evaluator'] : []),
      ]
      await createPrompt.mutateAsync({
        name: formData.name,
        description: formData.description || undefined,
        prompt_text: formData.prompt_text,
        prompt_type: formData.prompt_type,
        target: formData.target || undefined,
        category: formData.category || undefined,
        tags: finalTags.length > 0 ? finalTags : undefined,
      })
      toast.success('Prompt created successfully')
      router.push('/prompts')
    } catch (error) {
      console.error('Failed to create prompt:', error)
      toast.error('Failed to create prompt')
    }
  }

  return (
    <Layout>
      <div className="space-y-6 max-w-3xl">
        <h1 className="text-3xl font-bold">Create New Prompt</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="rounded-lg border bg-card p-6 space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name *</label>
              <input
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder="Enter prompt name"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Description</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder="Enter prompt description (optional)"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt Type *</label>
              <select
                required
                value={formData.prompt_type}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    prompt_type: e.target.value,
                    target: e.target.value === 'test_prompt' ? '' : formData.target,
                  })
                }
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="test_prompt">Test Prompt</option>
                <option value="system_prompt">System Prompt</option>
              </select>
            </div>

            {formData.prompt_type === 'system_prompt' && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Target *</label>
                <select
                  required
                  value={formData.target}
                  onChange={(e) => setFormData({ ...formData, target: e.target.value })}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value="">Select target</option>
                  <option value="doctor">Doctor Model</option>
                  <option value="patient">Patient Model</option>
                  <option value="evaluator">Evaluator</option>
                </select>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt Text *</label>
              <textarea
                required
                value={formData.prompt_text}
                onChange={(e) => setFormData({ ...formData, prompt_text: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder={formData.prompt_type === 'system_prompt' ? 'Enter the system prompt text' : 'Enter the prompt text'}
                rows={10}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="">Select category (optional)</option>
                <option value="adversarial">Jailbreak Prompts (Adversarial)</option>
                <option value="conversation">Conversation</option>
                <option value="scenario">Scenario</option>
                <option value="reasoning">Reasoning</option>
                <option value="safety">Safety</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Tags</label>
              <input
                type="text"
                value={formData.tags}
                onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                placeholder="Comma-separated tags (e.g., jailbreak, safety, test)"
              />
            </div>
          </div>

          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => router.back()}
              className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={
                createPrompt.isPending || 
                !formData.name || 
                !formData.prompt_text ||
                (formData.prompt_type === 'system_prompt' && !formData.target)
              }
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {createPrompt.isPending ? 'Creating...' : 'Create Prompt'}
            </button>
          </div>
        </form>
      </div>
    </Layout>
  )
}
