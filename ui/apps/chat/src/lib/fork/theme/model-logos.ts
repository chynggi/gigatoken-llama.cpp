/**
 * Online logo sources for model families.
 * Icons are loaded from the public @lobehub/icons-static-svg CDN on jsDelivr.
 */

export const MODEL_LOGO_CDN_BASE =
	'https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@1.91.0/icons';

/** Known model families inferred from model name/path/org. */
export type ModelFamily =
	| 'qwen'
	| 'llama'
	| 'gemma'
	| 'mistral'
	| 'deepseek'
	| 'phi'
	| 'yi'
	| 'glm'
	| 'cohere'
	| 'moonshot'
	| 'grok'
	| 'baichuan'
	| 'internlm'
	| 'rwkv'
	| 'nvidia'
	| 'ibm'
	| 'google'
	| 'openai'
	| 'anthropic'
	| 'huggingface'
	| 'unknown';

export interface ModelFamilyDefinition {
	family: ModelFamily;
	/** Human label for tooltips */
	label: string;
	/** CDN icon filename without extension (prefer monochrome/base for dark+light) */
	icon: string;
	/** Optional color variant icon when available */
	colorIcon?: string;
}

/**
 * Ordered matchers: first match wins.
 * Patterns run against the full model id/path (case-insensitive).
 */
export const MODEL_FAMILY_MATCHERS: Array<{
	family: ModelFamily;
	pattern: RegExp;
}> = [
	// More specific / vendor tokens first (DeepSeek distill names often include Qwen/Llama)
	{ family: 'deepseek', pattern: /\bdeepseek\b/i },
	{ family: 'qwen', pattern: /\b(qwen|qwq|qvq|tongyi)\b/i },
	{ family: 'gemma', pattern: /\bgemma\b/i },
	{ family: 'mistral', pattern: /\b(mistral|mixtral|codestral|pixtral|ministral)\b/i },
	{ family: 'phi', pattern: /\bphi[-\s]?\d|\bphi\b/i },
	{ family: 'yi', pattern: /\b(yi[-\s]?\d|01-ai)\b/i },
	{ family: 'glm', pattern: /\b(glm[-\s]?\d|chatglm|zhipu|thudm)\b/i },
	{ family: 'cohere', pattern: /\b(command[-\s]?r|cohere|aya)\b/i },
	{ family: 'moonshot', pattern: /\b(moonshot|kimi)\b/i },
	{ family: 'grok', pattern: /\b(grok|xai)\b/i },
	{ family: 'baichuan', pattern: /\bbaichuan\b/i },
	{ family: 'internlm', pattern: /\b(internlm|internvl)\b/i },
	{ family: 'rwkv', pattern: /\brwkv\b/i },
	{ family: 'nvidia', pattern: /\b(nemotron|nvidia)\b/i },
	{ family: 'ibm', pattern: /\b(granite|ibm)\b/i },
	{ family: 'llama', pattern: /\b(llama|meta-llama|codellama|tinyllama)\b/i },
	{ family: 'google', pattern: /\b(gemini|google\/gemma)\b/i },
	{ family: 'openai', pattern: /\b(gpt[-\s]?\d|openai|o1|o3|o4)\b/i },
	{ family: 'anthropic', pattern: /\b(claude|anthropic)\b/i },
	{ family: 'huggingface', pattern: /\b(smollm|hugging(?:face)?)\b/i }
];

/**
 * Map GGUF `general.architecture` values (and common prefixes) to logo families.
 * Exact keys first; otherwise the longest matching prefix in GGUF_ARCH_PREFIX_MAP wins.
 */
export const GGUF_ARCH_FAMILY_MAP: Record<string, ModelFamily> = {
	// Llama family
	llama: 'llama',
	llama4: 'llama',
	'llama-embed': 'llama',
	// Qwen family
	qwen: 'qwen',
	qwen2: 'qwen',
	qwen2moe: 'qwen',
	qwen2vl: 'qwen',
	qwen3: 'qwen',
	qwen3moe: 'qwen',
	qwen3next: 'qwen',
	qwen3vl: 'qwen',
	qwen3vlmoe: 'qwen',
	qwen35: 'qwen',
	qwen35moe: 'qwen',
	// Gemma
	gemma: 'gemma',
	gemma2: 'gemma',
	gemma3: 'gemma',
	gemma3n: 'gemma',
	gemma4: 'gemma',
	'gemma4-assistant': 'gemma',
	'gemma-embedding': 'gemma',
	// Phi / Microsoft
	phi2: 'phi',
	phi3: 'phi',
	phimoe: 'phi',
	// DeepSeek
	deepseek: 'deepseek',
	deepseek2: 'deepseek',
	'deepseek2-ocr': 'deepseek',
	deepseek32: 'deepseek',
	deepseek4: 'deepseek',
	// Mistral
	mistral3: 'mistral',
	mistral4: 'mistral',
	// GLM / ChatGLM
	chatglm: 'glm',
	glm4: 'glm',
	glm4moe: 'glm',
	'glm-dsa': 'glm',
	// Cohere
	'command-r': 'cohere',
	cohere2: 'cohere',
	cohere2moe: 'cohere',
	// IBM Granite
	granite: 'ibm',
	granitemoe: 'ibm',
	granitehybrid: 'ibm',
	// NVIDIA Nemotron
	nemotron: 'nvidia',
	nemotron_h: 'nvidia',
	nemotron_h_moe: 'nvidia',
	// RWKV
	rwkv6: 'rwkv',
	rwkv6qwen2: 'rwkv',
	rwkv7: 'rwkv',
	arwkv7: 'rwkv',
	// Others
	baichuan: 'baichuan',
	internlm2: 'internlm',
	grok: 'grok',
	'kimi-linear': 'moonshot',
	'gpt-oss': 'openai',
	smollm3: 'huggingface',
	gpt2: 'openai',
	gptj: 'openai',
	gptneox: 'openai'
};

/** Prefix fallbacks when exact GGUF arch is not listed above */
export const GGUF_ARCH_PREFIX_MAP: Array<{ prefix: string; family: ModelFamily }> = [
	{ prefix: 'qwen', family: 'qwen' },
	{ prefix: 'llama', family: 'llama' },
	{ prefix: 'gemma', family: 'gemma' },
	{ prefix: 'phi', family: 'phi' },
	{ prefix: 'deepseek', family: 'deepseek' },
	{ prefix: 'mistral', family: 'mistral' },
	{ prefix: 'glm', family: 'glm' },
	{ prefix: 'chatglm', family: 'glm' },
	{ prefix: 'cohere', family: 'cohere' },
	{ prefix: 'command', family: 'cohere' },
	{ prefix: 'granite', family: 'ibm' },
	{ prefix: 'nemotron', family: 'nvidia' },
	{ prefix: 'rwkv', family: 'rwkv' },
	{ prefix: 'arwkv', family: 'rwkv' },
	{ prefix: 'baichuan', family: 'baichuan' },
	{ prefix: 'internlm', family: 'internlm' },
	{ prefix: 'kimi', family: 'moonshot' },
	{ prefix: 'gpt', family: 'openai' },
	{ prefix: 'smollm', family: 'huggingface' }
];

/** HF org prefixes -> family when filename is ambiguous */
export const MODEL_ORG_FAMILY_MAP: Record<string, ModelFamily> = {
	'meta-llama': 'llama',
	qwen: 'qwen',
	'qwenlm': 'qwen',
	google: 'gemma',
	mistralai: 'mistral',
	microsoft: 'phi',
	'deepseek-ai': 'deepseek',
	'01-ai': 'yi',
	thudm: 'glm',
	'zai-org': 'glm',
	cohereai: 'cohere',
	'cohereforai': 'cohere',
	moonshotai: 'moonshot',
	'xai-org': 'grok',
	'baichuan-inc': 'baichuan',
	internlm: 'internlm',
	openai: 'openai',
	anthropic: 'anthropic',
	'ibm-granite': 'ibm',
	nvidia: 'nvidia',
	huggingface: 'huggingface',
	HuggingFaceTB: 'huggingface'
};

export const MODEL_FAMILY_DEFINITIONS: Record<ModelFamily, ModelFamilyDefinition> = {
	qwen: { family: 'qwen', label: 'Qwen', icon: 'qwen', colorIcon: 'qwen' },
	llama: { family: 'llama', label: 'Llama', icon: 'meta', colorIcon: 'meta-color' },
	gemma: { family: 'gemma', label: 'Gemma', icon: 'gemma', colorIcon: 'gemma-color' },
	mistral: { family: 'mistral', label: 'Mistral', icon: 'mistral', colorIcon: 'mistral' },
	deepseek: { family: 'deepseek', label: 'DeepSeek', icon: 'deepseek', colorIcon: 'deepseek' },
	phi: { family: 'phi', label: 'Phi', icon: 'microsoft', colorIcon: 'microsoft-color' },
	yi: { family: 'yi', label: 'Yi', icon: 'yi', colorIcon: 'yi' },
	glm: { family: 'glm', label: 'GLM', icon: 'chatglm', colorIcon: 'chatglm-color' },
	cohere: { family: 'cohere', label: 'Cohere', icon: 'cohere', colorIcon: 'cohere' },
	moonshot: { family: 'moonshot', label: 'Moonshot / Kimi', icon: 'kimi', colorIcon: 'kimi' },
	grok: { family: 'grok', label: 'Grok', icon: 'grok', colorIcon: 'xai' },
	baichuan: { family: 'baichuan', label: 'Baichuan', icon: 'baichuan', colorIcon: 'baichuan' },
	internlm: { family: 'internlm', label: 'InternLM', icon: 'internlm', colorIcon: 'internlm' },
	rwkv: { family: 'rwkv', label: 'RWKV', icon: 'rwkv', colorIcon: 'rwkv-color' },
	nvidia: { family: 'nvidia', label: 'NVIDIA', icon: 'nvidia', colorIcon: 'nvidia-color' },
	ibm: { family: 'ibm', label: 'IBM Granite', icon: 'ibm', colorIcon: 'ibm' },
	google: { family: 'google', label: 'Google', icon: 'google', colorIcon: 'google' },
	openai: { family: 'openai', label: 'OpenAI', icon: 'openai', colorIcon: 'openai' },
	anthropic: { family: 'anthropic', label: 'Anthropic', icon: 'claude', colorIcon: 'claude' },
	huggingface: {
		family: 'huggingface',
		label: 'Hugging Face',
		icon: 'huggingface',
		colorIcon: 'huggingface-color'
	},
	unknown: { family: 'unknown', label: 'Model', icon: 'ollama', colorIcon: 'ollama' }
};

export function modelLogoUrl(iconName: string): string {
	return `${MODEL_LOGO_CDN_BASE}/${iconName}.svg`;
}
