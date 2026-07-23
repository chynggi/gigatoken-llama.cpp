<script lang="ts">
	import {
		ChatMessageAgenticContent,
		ChatMessageActionIcons,
		ChatMessageAssistantModel,
		ChatMessageAssistantProcessingInfo,
		ChatMessageAssistantRawOutput,
		ChatMessageAssistantStatistics,
		ChatMessageEditForm,
		ModelLogo
	} from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { modelLoadProgressText } from '$lib/utils';
	import { MessageRole } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { SETTINGS_KEYS } from '$lib/constants';

	import { hasAgenticContent } from '$lib/utils';

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		showDeleteDialog: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		deletionInfo,
		isLastAssistantMessage = false,
		message,
		toolMessages = [],
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onForkConversation,
		onNavigateToSibling,
		onRegenerate,
		onShowDeleteDialogChange,
		showDeleteDialog,
		siblingInfo = null,
		textareaElement = $bindable()
	}: Props = $props();

	const editCtx = getMessageEditContext();

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const processingState = useProcessingState();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());

	let showRawOutput = $state(false);
	let showModelResponseLogo = $derived(
		Boolean(currentConfig[SETTINGS_KEYS.SHOW_MODEL_RESPONSE_LOGO] ?? true)
	);

	let displayedModel = $derived(message.model ?? null);
	let logoModelHint = $derived(
		displayedModel ??
			modelsStore.selectedModelName ??
			modelsStore.singleModelName ??
			serverStore.props?.model_path ??
			null
	);
	let logoArchitecture = $derived.by(() => {
		const fromProps = serverStore.props?.model_architecture;
		if (fromProps) return fromProps;

		const modelKey = displayedModel ?? modelsStore.selectedModelName;
		if (!modelKey || !isRouter) return null;

		const entry = modelsStore.routerModels.find(
			(m) => m.id === modelKey || m.name === modelKey || m.aliases?.includes(modelKey)
		);
		const arch = entry?.meta?.architecture;
		return typeof arch === 'string' && arch.length > 0 ? arch : null;
	});
	let logoChatTemplate = $derived(serverStore.props?.chat_template ?? null);
	let logoModelAlias = $derived(serverStore.props?.model_alias ?? null);

	let isCurrentlyLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);

	let loadTargetModel = $derived(message.model ?? modelsStore.selectedModelName);
	let modelLoadProgress = $derived(
		isRouter && loadTargetModel ? modelsStore.getLoadProgress(loadTargetModel) : null
	);
	let modelLoadingText = $derived(modelLoadProgressText(modelLoadProgress));

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	let assistantEl: HTMLDivElement | undefined = $state();
	let lastUserMessageHeight = $state(0);
	let assistantMarginTop = $state(0);

	$effect(() => {
		if (!assistantEl) return;

		assistantMarginTop = Math.round(parseFloat(getComputedStyle(assistantEl).marginTop));

		const chatMessageEl = assistantEl.closest('.chat-message');
		const previousChatMessage = chatMessageEl?.previousElementSibling;
		const userMessageEl = previousChatMessage?.querySelector(
			'.chat-message-user'
		) as HTMLElement | null;

		if (!userMessageEl) {
			lastUserMessageHeight = 0;
			return;
		}

		const updateHeight = () => {
			const rect = userMessageEl.getBoundingClientRect();
			const marginTop = Math.round(parseFloat(getComputedStyle(userMessageEl).marginTop));
			lastUserMessageHeight = Math.round(rect.height + marginTop);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);
		resizeObserver.observe(userMessageEl);

		return () => {
			resizeObserver.disconnect();
		};
	});

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	bind:this={assistantEl}
	class="chat-message-assistant text-md group w-full leading-7.5 {className} {isLastAssistantMessage &&
	isChatStreaming()
		? 'glass-glow-pulse rounded-2xl'
		: ''}"
	style:--last-user-message-height={lastUserMessageHeight > 0
		? `${lastUserMessageHeight}px`
		: undefined}
	style:--assistant-margin-top={assistantMarginTop > 0 ? `${assistantMarginTop}px` : undefined}
	role="group"
	aria-label="Assistant message with actions"
>
	{#if showProcessingInfoTop}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} {processingState} position="top" />
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<div class="flex w-full items-start gap-3">
			{#if showModelResponseLogo}
				<ModelLogo
					class="mt-1"
					model={logoModelHint}
					architecture={logoArchitecture}
					chatTemplate={logoChatTemplate}
					modelAlias={logoModelAlias}
					size="md"
				/>
			{/if}

			<div class="min-w-0 flex-1">
				{#if showRawOutput}
					<ChatMessageAssistantRawOutput {message} {toolMessages} />
				{:else}
					<ChatMessageAgenticContent
						{message}
						{toolMessages}
						isStreaming={isChatStreaming()}
						{isLastAssistantMessage}
					/>
				{/if}
			</div>
		</div>
	{/if}

	{#if showProcessingInfoBottom}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} {processingState} position="bottom" />
	{/if}

	<div class="info my-6 grid gap-4 tabular-nums">
		{#if displayedModel}
			<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
				<ChatMessageAssistantModel
					{displayedModel}
					isLoading={isLoading()}
					{isRouter}
					{onRegenerate}
				/>

				<ChatMessageAssistantStatistics
					{message}
					isLoading={isLoading()}
					{processingState}
					showMessageStats={currentConfig.showMessageStats}
				/>
			</div>
		{/if}
	</div>

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActionIcons
			role={MessageRole.ASSISTANT}
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
			{onForkConversation}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
			rawOutputEnabled={showRawOutput}
			onRawOutputToggle={(enabled) => (showRawOutput = enabled)}
		/>
	{/if}
</div>

<style>
	:global(.chat-message):last-child .chat-message-assistant {
		--assistant-min-height-offset: calc(
			var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
				var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
				var(--assistant-margin-top, 3rem)
		);
		min-height: calc(100dvh - var(--assistant-min-height-offset));

		@media (width > 768px) {
			--assistant-min-height-offset: calc(
				var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
					var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
					var(--assistant-margin-top, 3rem)
			);
		}
	}

	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}

	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}

	@media (min-width: 1536px) {
		:global(html.wide-chat-mode) .raw-output {
			max-width: 64rem;
		}
		:global(html.full-chat-mode) .raw-output {
			max-width: 100%;
		}
	}
</style>
