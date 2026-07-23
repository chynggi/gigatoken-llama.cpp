import { NEWLINE } from '$lib/constants';

/**
 * Returns a shortened preview of the provided content capped at the given length.
 * Appends an ellipsis when the content exceeds the maximum.
 */
export function getPreviewText(content: string, max = 150): string {
	return content.length > max ? content.slice(0, max) + '...' : content;
}

/**
 * Generates a single-line title from a potentially multi-line prompt.
 * Uses the first non-empty line if `useFirstLine` is true.
 */
export function generateConversationTitle(content: string, useFirstLine: boolean = false): string {
	if (useFirstLine) {
		const firstLine = content.split(NEWLINE).find((line) => line.trim().length > 0);
		return firstLine ? firstLine.trim() : content.trim();
	}

	return content.trim();
}

const REASONING_PREVIEW_MAX = 120;

export function formatReasoningPreview(content: string): { preview: string; overflow: number } {
	const trimmed = content.trim();
	if (trimmed.length === 0) return { preview: '', overflow: 0 };

	if (trimmed.length <= REASONING_PREVIEW_MAX) {
		return { preview: trimmed, overflow: 0 };
	}

	return {
		preview: trimmed.slice(0, REASONING_PREVIEW_MAX),
		overflow: trimmed.length - REASONING_PREVIEW_MAX
	};
}
