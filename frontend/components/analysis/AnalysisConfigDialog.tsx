import { useState } from 'react'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { getSettings } from '@/lib/settings'

interface AnalysisConfigDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: (config: AnalysisConfig) => void
  defaultConfig?: Partial<AnalysisConfig>
  doctorProvider?: string
  doctorModel?: string
  isLoading?: boolean
}

export interface AnalysisConfig {
  evaluator_provider?: string
  evaluator_model?: string
  enable_cot_detection: boolean
  cot_analysis_mode: 'full' | 'partial' | 'none'
  enable_factuality_check: boolean
  enable_manipulation_analysis: boolean
}

export function AnalysisConfigDialog({
  isOpen,
  onClose,
  onConfirm,
  defaultConfig,
  doctorProvider,
  doctorModel,
  isLoading = false,
}: AnalysisConfigDialogProps) {
  const savedSettings = getSettings()
  // When doctorProvider/doctorModel are provided, default to empty so the backend uses the doctor model.
  // Otherwise use saved settings (e.g. for batch analysis).
  const [evaluatorProvider, setEvaluatorProvider] = useState(
    defaultConfig?.evaluator_provider ?? (doctorProvider ? '' : savedSettings.defaultEvaluatorProvider)
  )
  const [evaluatorModel, setEvaluatorModel] = useState(
    defaultConfig?.evaluator_model ?? (doctorModel ? '' : savedSettings.defaultEvaluatorModel)
  )
  const [enableCotDetection, setEnableCotDetection] = useState(
    defaultConfig?.enable_cot_detection ?? savedSettings.defaultEnableCotDetection
  )
  const [cotAnalysisMode, setCotAnalysisMode] = useState<'full' | 'partial' | 'none'>(
    defaultConfig?.cot_analysis_mode ?? savedSettings.defaultCotAnalysisMode
  )
  const [enableFactualityCheck, setEnableFactualityCheck] = useState(
    defaultConfig?.enable_factuality_check ?? savedSettings.defaultEnableFactualityCheck
  )
  const [enableManipulationAnalysis, setEnableManipulationAnalysis] = useState(
    defaultConfig?.enable_manipulation_analysis ?? savedSettings.defaultEnableManipulationAnalysis
  )

  if (!isOpen) return null

  const handleConfirm = () => {
    const config = {
      evaluator_provider: evaluatorProvider || undefined,
      evaluator_model: evaluatorModel || undefined,
      enable_cot_detection: enableCotDetection,
      cot_analysis_mode: cotAnalysisMode,
      enable_factuality_check: enableFactualityCheck,
      enable_manipulation_analysis: enableManipulationAnalysis,
    }
    console.log('AnalysisConfigDialog: Confirming with config:', config)
    onConfirm(config)
    onClose()
  }

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          console.log('Dialog backdrop clicked, closing')
          onClose()
        }
      }}
    >
      <div 
        className="w-full max-w-2xl max-h-[90vh] rounded-lg border bg-card shadow-lg flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header - Fixed */}
        <div className="px-6 pt-6 pb-4 border-b flex-shrink-0">
          <h2 className="text-2xl font-semibold">Configure Analysis</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose which AI model will analyze the conversation and chain of thought reasoning.
          </p>
        </div>

        {/* Scrollable Content */}
        <div className="px-6 py-6 overflow-y-auto flex-1">
          <div className="space-y-6">
          {/* Evaluator Model Selection */}
          <div className="rounded-lg border bg-muted/30 p-4">
            <h3 className="mb-4 font-medium">Evaluator AI Model</h3>
            <p className="mb-4 text-sm text-muted-foreground">
              Select an AI model to analyze the conversation. If not specified, the doctor model will be used.
            </p>
            {doctorProvider && doctorModel && (
              <p className="mb-4 text-xs text-muted-foreground">
                Current doctor model: <span className="font-medium">{doctorModel}</span> ({doctorProvider})
              </p>
            )}
            <ModelSelector
              label=""
              provider={evaluatorProvider}
              model={evaluatorModel}
              onProviderChange={setEvaluatorProvider}
              onModelChange={setEvaluatorModel}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              Leave empty to use the doctor model as the evaluator
            </p>
          </div>

          {/* Chain of Thought Detection */}
          <div className="rounded-lg border bg-muted/30 p-4">
            <h3 className="mb-4 font-medium">Chain of Thought Analysis</h3>
            <div className="space-y-4">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={enableCotDetection}
                  onChange={(e) => setEnableCotDetection(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <div>
                  <span className="font-medium">Enable Chain of Thought Detection</span>
                  <p className="text-xs text-muted-foreground">
                    Detect and analyze chain of thought reasoning patterns in the conversation
                  </p>
                </div>
              </label>

              {enableCotDetection && (
                <div>
                  <label className="mb-2 block text-sm font-medium">Analysis Mode</label>
                  <select
                    value={cotAnalysisMode}
                    onChange={(e) => setCotAnalysisMode(e.target.value as 'full' | 'partial' | 'none')}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <option value="full">Full Analysis - Complete chain of thought detection</option>
                    <option value="partial">Partial Analysis - Basic pattern detection</option>
                    <option value="none">None - Disable COT analysis</option>
                  </select>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {cotAnalysisMode === 'full' &&
                      'Comprehensive analysis of reasoning chains and thought processes'}
                    {cotAnalysisMode === 'partial' &&
                      'Basic detection of reasoning indicators and patterns'}
                    {cotAnalysisMode === 'none' && 'No chain of thought analysis will be performed'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Factuality and Hallucination Detection */}
          <div className="rounded-lg border bg-muted/30 p-4">
            <h3 className="mb-4 font-medium">Factuality Analysis</h3>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={enableFactualityCheck}
                onChange={(e) => setEnableFactualityCheck(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300"
              />
              <div>
                <span className="font-medium">Enable Factuality/Hallucination Detection</span>
                <p className="text-xs text-muted-foreground">
                  Verify if responses are factual and identify potential hallucinations. Analyzes patient model responses for accuracy and truthfulness.
                </p>
              </div>
            </label>
          </div>

          {/* Manipulation Analysis */}
          <div className="rounded-lg border bg-muted/30 p-4">
            <h3 className="mb-4 font-medium">Manipulation Analysis</h3>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={enableManipulationAnalysis}
                onChange={(e) => setEnableManipulationAnalysis(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300"
              />
              <div>
                <span className="font-medium">Enable Manipulation Analysis</span>
                <p className="text-xs text-muted-foreground">
                  Analyze manipulation resistance (how easily the patient model can be manipulated) and manipulation capability (how well the doctor model manipulates others).
                </p>
              </div>
            </label>
          </div>
          </div>
        </div>

        {/* Footer - Fixed */}
        <div className="px-6 py-4 border-t flex justify-end gap-3 flex-shrink-0">
          <button
            onClick={onClose}
            className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
          >
            Cancel
          </button>
          <button
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              console.log('Dialog: Run Analysis button clicked')
              handleConfirm()
            }}
            disabled={isLoading}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Starting...' : 'Run Analysis'}
          </button>
        </div>
      </div>
    </div>
  )
}
