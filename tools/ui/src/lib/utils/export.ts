/**
 * Export utilities - Markdown and HTML export for conversations.
 *
 * Converts a conversation + messages into a clean Markdown or HTML document
 * suitable for sharing, printing, or archiving.
 */

import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

/**
 * Convert a conversation and its messages to a Markdown string.
 */
export function exportConversationAsMarkdown(
	conv: DatabaseConversation,
	messages: DatabaseMessage[]
): string {
	const lines: string[] = [];

	// Header
	lines.push(`# ${conv.name || 'Conversation'}`);
	lines.push('');
	lines.push(`_Exported on ${new Date().toLocaleString()}_`);
	lines.push('');

	// Messages
	for (const msg of messages) {
		if (msg.type === 'root') continue;

		const role = msg.role;

		const roleLabel = role === 'user' ? '### You' : role === 'assistant' ? '### Assistant' : role === 'system' ? '### System' : `### ${role}`;

		lines.push(roleLabel);
		lines.push('');

		if (msg.reasoningContent) {
			lines.push('<details>');
			lines.push('<summary>Thinking</summary>');
			lines.push('');
			lines.push(msg.reasoningContent);
			lines.push('');
			lines.push('</details>');
			lines.push('');
		}

		lines.push(msg.content);
		lines.push('');

		if (msg.extra && msg.extra.length > 0) {
			for (const extra of msg.extra) {
				if ('name' in extra) {
					lines.push(`_Attachment: ${extra.name}_`);
					lines.push('');
				}
			}
		}

		// Separator
		lines.push('---');
		lines.push('');
	}

	return lines.join('\n');
}

/**
 * Convert a conversation and its messages to a self-contained HTML document.
 */
export function exportConversationAsHtml(
	conv: DatabaseConversation,
	messages: DatabaseMessage[]
): string {
	const bodyLines: string[] = [];

	// Header
	bodyLines.push(`<h1>${escapeHtml(conv.name || 'Conversation')}</h1>`);
	bodyLines.push(`<p class="export-date">Exported on ${new Date().toLocaleString()}</p>`);

	// Messages
	for (const msg of messages) {
		if (msg.type === 'root') continue;

		const roleClass = msg.role === 'user' ? 'user' : msg.role === 'assistant' ? 'assistant' : msg.role === 'system' ? 'system' : 'other';
		bodyLines.push(`<div class="message ${roleClass}">`);
		bodyLines.push(`<div class="role">${escapeHtml(msg.role)}</div>`);

		if (msg.reasoningContent) {
			bodyLines.push('<details class="reasoning">');
			bodyLines.push('<summary>Thinking</summary>');
			bodyLines.push(`<pre>${escapeHtml(msg.reasoningContent)}</pre>`);
			bodyLines.push('</details>');
		}

		bodyLines.push(`<div class="content">${escapeHtml(msg.content).replace(/\n/g, '<br>')}</div>`);

		if (msg.extra && msg.extra.length > 0) {
			for (const extra of msg.extra) {
				if ('name' in extra) {
					bodyLines.push(`<div class="attachment">Attachment: ${escapeHtml(extra.name)}</div>`);
				}
			}
		}

		bodyLines.push('</div>');
	}

	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(conv.name || 'Conversation')}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.5rem; border-bottom: 2px solid #e5e5e5; padding-bottom: 0.5rem; }
.export-date { color: #666; font-size: 0.85rem; }
.message { margin: 1.5rem 0; padding: 1rem; border-radius: 8px; }
.message.user { background: #f0f4ff; }
.message.assistant { background: #f5f5f5; }
.message.system { background: #fffbe6; border-left: 3px solid #f0c000; }
.role { font-weight: 600; font-size: 0.8rem; text-transform: uppercase; color: #888; margin-bottom: 0.5rem; }
.reasoning { margin-bottom: 0.75rem; }
.reasoning pre { background: #fafafa; padding: 0.75rem; border-radius: 4px; font-size: 0.85rem; white-space: pre-wrap; }
.content { white-space: pre-wrap; }
.attachment { font-size: 0.8rem; color: #888; margin-top: 0.5rem; font-style: italic; }
</style>
</head>
<body>
${bodyLines.join('\n')}
</body>
</html>`;
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;');
}
