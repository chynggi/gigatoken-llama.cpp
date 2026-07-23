import { describe, expect, it } from 'vitest';
import { familyFromGgufArchitecture, parseModelFamily } from '$lib/utils/model-family';

describe('familyFromGgufArchitecture', () => {
	it('maps exact GGUF arch strings', () => {
		expect(familyFromGgufArchitecture('qwen35moe')).toBe('qwen');
		expect(familyFromGgufArchitecture('gemma4')).toBe('gemma');
		expect(familyFromGgufArchitecture('llama')).toBe('llama');
		expect(familyFromGgufArchitecture('deepseek2')).toBe('deepseek');
		expect(familyFromGgufArchitecture('phi3')).toBe('phi');
	});

	it('maps via prefix when exact key is missing', () => {
		expect(familyFromGgufArchitecture('qwen99-experimental')).toBe('qwen');
	});
});

describe('parseModelFamily', () => {
	it('prefers GGUF architecture over misleading file names', () => {
		const r = parseModelFamily('my-finetune-v3.gguf', { architecture: 'qwen35moe' });
		expect(r.family).toBe('qwen');
		expect(r.source).toBe('gguf-architecture');
		expect(r.ggufArchitecture).toBe('qwen35moe');
	});

	it('detects Qwen from model id', () => {
		const r = parseModelFamily('Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M');
		expect(r.family).toBe('qwen');
		expect(r.logoUrl).toContain('qwen');
	});

	it('detects Llama from meta-llama path', () => {
		const r = parseModelFamily('meta-llama/Llama-3.1-8B-Instruct');
		expect(r.family).toBe('llama');
		expect(r.logoUrl).toContain('meta');
	});

	it('detects Gemma from filename', () => {
		const r = parseModelFamily('/models/gemma-3-4b-it-Q4_K_M.gguf');
		expect(r.family).toBe('gemma');
	});

	it('detects DeepSeek', () => {
		const r = parseModelFamily('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B');
		// deepseek matcher is ordered before qwen; should win
		expect(r.family).toBe('deepseek');
	});

	it('detects Phi via microsoft org', () => {
		const r = parseModelFamily('microsoft/Phi-3-mini-4k-instruct');
		expect(r.family).toBe('phi');
	});

	it('uses chat template as a fallback hint', () => {
		const r = parseModelFamily('local-custom-model.bin', {
			chatTemplate: '{% if messages %}qwen{% endif %}'
		});
		expect(r.family).toBe('qwen');
		expect(r.source).toBe('chat-template');
	});

	it('returns unknown for opaque names', () => {
		const r = parseModelFamily('my-finetune-v3');
		expect(r.family).toBe('unknown');
		expect(r.logoUrl).toContain('ollama');
	});
});
