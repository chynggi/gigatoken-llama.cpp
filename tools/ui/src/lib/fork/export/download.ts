/**
 * Browser glue for the fork's Markdown / HTML conversation export.
 *
 * The formatting itself is pure and lives in `export.ts`; everything that needs
 * the DOM or the database sits here so the formatters stay testable.
 */

import { EXPORT_CONV } from '$lib/constants';
import { exportConversationAsHtml, exportConversationAsMarkdown } from '$lib/fork/export/export';
import { DatabaseService } from '$lib/services/database.service';

/** Turn a conversation name into a safe download filename stem. */
function toFilenameStem(name: string): string {
	return (
		name.replace(EXPORT_CONV.NON_ALPHANUMERIC_REGEX, EXPORT_CONV.NONALNUM_REPLACEMENT) ||
		'conversation'
	);
}

/**
 * Hand a blob to the browser as a download.
 *
 * Upstream has an equivalent private helper on `conversationsStore`; duplicating
 * these few lines keeps the fork from widening an upstream API.
 */
function triggerDownload(blob: Blob, filename: string): void {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');

	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

/** Load a conversation and its messages, or `null` when it no longer exists. */
async function loadExportData(convId: string) {
	const conv = await DatabaseService.getConversation(convId);

	if (!conv) return null;

	const messages = await DatabaseService.getConversationMessages(convId);

	return { conv, messages };
}

/**
 * Download a conversation as a Markdown document.
 *
 * @param convId - Conversation to export; a missing conversation is a no-op
 */
export async function downloadConversationMarkdown(convId: string): Promise<void> {
	const data = await loadExportData(convId);

	if (!data) return;

	const markdown = exportConversationAsMarkdown(data.conv, data.messages);
	const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });

	triggerDownload(blob, `${toFilenameStem(data.conv.name)}.md`);
}

/**
 * Download a conversation as a self-contained HTML document.
 *
 * @param convId - Conversation to export; a missing conversation is a no-op
 */
export async function downloadConversationHtml(convId: string): Promise<void> {
	const data = await loadExportData(convId);

	if (!data) return;

	const html = exportConversationAsHtml(data.conv, data.messages);
	const blob = new Blob([html], { type: 'text/html' });

	triggerDownload(blob, `${toFilenameStem(data.conv.name)}.html`);
}
