import {
	GGUF_ARCH_FAMILY_MAP,
	GGUF_ARCH_PREFIX_MAP,
	MODEL_FAMILY_DEFINITIONS,
	MODEL_FAMILY_MATCHERS,
	MODEL_ORG_FAMILY_MAP,
	modelLogoUrl,
	type ModelFamily,
	type ModelFamilyDefinition
} from '$lib/constants/model-logos';

export interface ResolvedModelFamily {
	family: ModelFamily;
	definition: ModelFamilyDefinition;
	/** Preferred logo URL (color variant when available) */
	logoUrl: string;
	/** Monochrome / base logo URL */
	logoUrlMono: string;
	/** Source used for resolution */
	source: 'gguf-architecture' | 'name' | 'org' | 'chat-template' | 'unknown';
	/** Raw GGUF architecture string when provided */
	ggufArchitecture?: string;
}

export interface ParseModelFamilyHints {
	/** GGUF general.architecture from server props / model meta (preferred) */
	architecture?: string | null;
	chatTemplate?: string | null;
	modelAlias?: string | null;
}

/**
 * Map a GGUF `general.architecture` value to a logo family.
 * Prefers exact match, then longest prefix match.
 */
export function familyFromGgufArchitecture(architecture?: string | null): ModelFamily | null {
	if (!architecture) return null;

	const arch = architecture.trim().toLowerCase();
	if (!arch || arch === '(unknown)') return null;

	const exact = GGUF_ARCH_FAMILY_MAP[arch];
	if (exact) return exact;

	let best: { prefix: string; family: ModelFamily } | null = null;
	for (const entry of GGUF_ARCH_PREFIX_MAP) {
		if (arch === entry.prefix || arch.startsWith(entry.prefix)) {
			if (!best || entry.prefix.length > best.prefix.length) {
				best = entry;
			}
		}
	}

	return best?.family ?? null;
}

/**
 * Infer model family for response logos.
 * Priority:
 * 1. GGUF general.architecture (server-provided)
 * 2. Model id / path / alias name tokens
 * 3. HF org prefix
 * 4. Chat template fingerprint
 */
export function parseModelFamily(
	modelIdOrPath?: string | null,
	hints?: ParseModelFamilyHints
): ResolvedModelFamily {
	const ggufArch = hints?.architecture?.trim() || '';

	// 1) Authoritative: GGUF metadata architecture
	if (ggufArch) {
		const fromArch = familyFromGgufArchitecture(ggufArch);
		if (fromArch) {
			return resolveFamily(fromArch, 'gguf-architecture', ggufArch);
		}
	}

	const samples = [modelIdOrPath, hints?.modelAlias]
		.filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
		.map((v) => v.trim());

	const nameHaystack = samples.join('\n');

	// 2) Token patterns against model id / path / alias
	if (nameHaystack) {
		for (const matcher of MODEL_FAMILY_MATCHERS) {
			if (matcher.pattern.test(nameHaystack)) {
				return resolveFamily(matcher.family, 'name', ggufArch || undefined);
			}
		}
	}

	// 3) HF org prefix: org/model
	for (const sample of samples) {
		const org = extractOrg(sample);
		if (!org) continue;

		const mapped = MODEL_ORG_FAMILY_MAP[org] ?? MODEL_ORG_FAMILY_MAP[org.toLowerCase()];
		if (mapped) {
			return resolveFamily(mapped, 'org', ggufArch || undefined);
		}
	}

	// 4) Chat template as last-resort hint
	const template = hints?.chatTemplate?.trim();
	if (template) {
		for (const matcher of MODEL_FAMILY_MATCHERS) {
			if (matcher.pattern.test(template)) {
				return resolveFamily(matcher.family, 'chat-template', ggufArch || undefined);
			}
		}
	}

	return resolveFamily('unknown', 'unknown', ggufArch || undefined);
}

function extractOrg(modelId: string): string | null {
	const cleaned = modelId.replace(/\\/g, '/');
	const parts = cleaned.split('/').filter(Boolean);

	if (parts.length >= 2) {
		const maybeOrg = parts[parts.length - 2];
		const maybeModel = parts[parts.length - 1];

		if (maybeOrg && maybeModel && !maybeOrg.includes('.')) {
			return maybeOrg;
		}
	}

	return null;
}

function resolveFamily(
	family: ModelFamily,
	source: ResolvedModelFamily['source'],
	ggufArchitecture?: string
): ResolvedModelFamily {
	const definition = MODEL_FAMILY_DEFINITIONS[family] ?? MODEL_FAMILY_DEFINITIONS.unknown;
	const mono = modelLogoUrl(definition.icon);
	const color = definition.colorIcon ? modelLogoUrl(definition.colorIcon) : mono;

	return {
		family: definition.family,
		definition,
		logoUrl: color,
		logoUrlMono: mono,
		source,
		ggufArchitecture
	};
}

export function getModelLogoUrl(
	modelIdOrPath?: string | null,
	hints?: ParseModelFamilyHints,
	preferColor = true
): string {
	const resolved = parseModelFamily(modelIdOrPath, hints);
	return preferColor ? resolved.logoUrl : resolved.logoUrlMono;
}
