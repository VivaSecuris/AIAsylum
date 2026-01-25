import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { usePrompt, useUpdatePrompt } from '@/lib/hooks'
import { toast } from '@/lib/toast'

export default function EditPromptPage() {
  const router = useRouter()
  const { id } = router.query
  const promptId = typeof id === 'string' ? parseInt(id) : 0
  const { data: prompt, isLoading } = usePrompt(promptId)
  const updatePrompt = useUpdatePrompt()

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    prompt_text: '',
    prompt_type: 'test_prompt',
    target: '',
    category: '',
    tags: '',
  })

  useEffect(() => {
    if (prompt) {
      setFormData({
        name: prompt.name,
        description: prompt.description || '',
        prompt_text: prompt.prompt_text,
        prompt_type: prompt.prompt_type || 'test_prompt',
        target: prompt.target || '',
        category: prompt.category || '',
        tags: prompt.tags?.join(', ') || '',
      })
    }
  }, [prompt])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!promptId) return
    try {
      const tags = formData.tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean)
      await updatePrompt.mutateAsync({
        id: promptId,
        data: {
          name: formData.name,
          description: formData.description || undefined,
          prompt_text: formData.prompt_text,
          prompt_type: formData.prompt_type,
          target: formData.target || undefined,
          category: formData.category || undefined,
          tags: tags.length > 0 ? tags : undefined,
        },
      })
      toast.success('Prompt updated successfully')
      router.push('/prompts')
    } catch (error) {
      console.error('Failed to update prompt:', error)
      toast.error('Failed to update prompt')
    }
  }

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  if (!prompt) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Prompt not found</h2>
            <p className="text-muted-foreground">The prompt you're looking for doesn't exist.</p>
          </div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6 max-w-3xl">
        <h1 className="text-3xl font-bold">Edit Prompt</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="rounded-lg border bg-card p-6 space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name *</label>
              <input
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm"
                placeholder="Enter prompt name"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Description</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm"
                placeholder="Enter prompt description (optional)"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt Type *</label>
              <select
                required
                value={formData.prompt_type}
                onChange={(e) => setFormData({ ...formData, prompt_type: e.target.value, target: e.target.value === 'test_prompt' ? '' : formData.target })}
                className="w-full rounded-lg border px-3 py-2 text-sm"
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
                  className="w-full rounded-lg border px-3 py-2 text-sm"
                >
                  <option value="">Select target</option>
                  <option value="doctor">Doctor Model</option>
                  <option value="patient">Patient Model</option>
                </select>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt Text *</label>
              <textarea
                required
                value={formData.prompt_text}
                onChange={(e) => setFormData({ ...formData, prompt_text: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm font-mono"
                placeholder={formData.prompt_type === 'system_prompt' ? 'Enter the system prompt text' : 'Enter the prompt text'}
                rows={10}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                className="w-full rounded-lg border px-3 py-2 text-sm"
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
                className="w-full rounded-lg border px-3 py-2 text-sm"
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
                updatePrompt.isPending || 
                !formData.name || 
                !formData.prompt_text ||
                (formData.prompt_type === 'system_prompt' && !formData.target)
              }
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {updatePrompt.isPending ? 'Updating...' : 'Update Prompt'}
            </button>
          </div>
        </form>
      </div>
    </Layout>
  )
}
