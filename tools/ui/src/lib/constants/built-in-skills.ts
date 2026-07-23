/**
 * Built-in skill templates seeded on first skill-store init.
 */

export const BUILT_IN_SKILLS: Omit<DatabaseSkill, 'id' | 'createdAt'>[] = [
	{
		name: 'summarize',
		description: 'Summarize the provided text or conversation',
		icon: 'FileText',
		content: 'Please summarize the following:\n\n{{text}}',
		category: 'writing',
		placeholders: [{ name: 'text', description: 'Text or content to summarize' }],
		isBuiltIn: true
	},
	{
		name: 'translate',
		description: 'Translate text to another language',
		icon: 'Languages',
		content: 'Translate the following text to {{lang}}:\n\n{{text}}',
		category: 'writing',
		placeholders: [
			{ name: 'lang', description: 'Target language (e.g. ko, en, ja)', defaultValue: 'en' },
			{ name: 'text', description: 'Text to translate' }
		],
		isBuiltIn: true
	},
	{
		name: 'rewrite',
		description: 'Rewrite text with a different style or tone',
		icon: 'PenLine',
		content: 'Rewrite the following text in a {{style}} style:\n\n{{text}}',
		category: 'writing',
		placeholders: [
			{
				name: 'style',
				description: 'Style: formal, casual, concise, professional',
				defaultValue: 'professional'
			},
			{ name: 'text', description: 'Text to rewrite' }
		],
		isBuiltIn: true
	},
	{
		name: 'email',
		description: 'Draft an email from a brief description',
		icon: 'Mail',
		content: 'Write a professional email about the following:\n\n{{topic}}\n\nTone: {{tone}}',
		category: 'writing',
		placeholders: [
			{ name: 'topic', description: 'What the email is about' },
			{ name: 'tone', description: 'Tone: formal, friendly, concise', defaultValue: 'formal' }
		],
		isBuiltIn: true
	},
	{
		name: 'blog-post',
		description: 'Generate a blog post from a topic',
		icon: 'Newspaper',
		content:
			'Write a blog post about "{{topic}}". Include an introduction, key points with subheadings, and a conclusion.',
		category: 'writing',
		placeholders: [{ name: 'topic', description: 'Blog post topic' }],
		isBuiltIn: true
	},
	{
		name: 'explain-code',
		description: 'Explain what a piece of code does',
		icon: 'Code',
		content: 'Explain the following code in detail:\n\n```\n{{code}}\n```',
		category: 'coding',
		placeholders: [{ name: 'code', description: 'Code to explain' }],
		isBuiltIn: true
	},
	{
		name: 'refactor',
		description: 'Suggest refactoring improvements for code',
		icon: 'Wrench',
		content:
			'Review the following code and suggest refactoring improvements for readability, performance, and maintainability:\n\n```\n{{code}}\n```',
		category: 'coding',
		placeholders: [{ name: 'code', description: 'Code to refactor' }],
		isBuiltIn: true
	},
	{
		name: 'add-tests',
		description: 'Generate unit tests for code',
		icon: 'FlaskConical',
		content: 'Write comprehensive unit tests for the following code:\n\n```\n{{code}}\n```',
		category: 'coding',
		placeholders: [{ name: 'code', description: 'Code to write tests for' }],
		isBuiltIn: true
	},
	{
		name: 'fix-bug',
		description: 'Help diagnose and fix a bug',
		icon: 'Bug',
		content:
			'Help me debug the following issue:\n\nError/Behavior: {{error}}\n\nCode:\n```\n{{code}}\n```',
		category: 'coding',
		placeholders: [
			{ name: 'error', description: 'Description of the error or unexpected behavior' },
			{ name: 'code', description: 'Relevant code' }
		],
		isBuiltIn: true
	},
	{
		name: 'compare',
		description: 'Compare two items or concepts',
		icon: 'GitCompare',
		content:
			'Compare and contrast {{a}} and {{b}}. Highlight key similarities, differences, and use cases for each.',
		category: 'analysis',
		placeholders: [
			{ name: 'a', description: 'First item to compare' },
			{ name: 'b', description: 'Second item to compare' }
		],
		isBuiltIn: true
	},
	{
		name: 'pros-cons',
		description: 'Analyze pros and cons of a decision',
		icon: 'Scale',
		content: 'Analyze the pros and cons of: {{decision}}',
		category: 'analysis',
		placeholders: [{ name: 'decision', description: 'Decision or option to analyze' }],
		isBuiltIn: true
	},
	{
		name: 'analyze-data',
		description: 'Analyze provided data and extract insights',
		icon: 'BarChart',
		content:
			'Analyze the following data and provide key insights, trends, and recommendations:\n\n{{data}}',
		category: 'analysis',
		placeholders: [{ name: 'data', description: 'Data to analyze (CSV, JSON, text, etc.)' }],
		isBuiltIn: true
	},
	{
		name: 'think-step-by-step',
		description: 'Break down a complex problem step by step',
		icon: 'ListOrdered',
		content: "Let's think through this step by step:\n\n{{problem}}",
		category: 'reasoning',
		placeholders: [{ name: 'problem', description: 'Problem to reason through' }],
		isBuiltIn: true
	},
	{
		name: 'brainstorm',
		description: 'Generate creative ideas for a topic',
		icon: 'Lightbulb',
		content:
			'Brainstorm creative ideas for: {{topic}}. Generate at least 5 diverse and innovative suggestions.',
		category: 'reasoning',
		placeholders: [{ name: 'topic', description: 'Topic to brainstorm about' }],
		isBuiltIn: true
	}
];
