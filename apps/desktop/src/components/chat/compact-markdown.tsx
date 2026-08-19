import type { ComponentProps, ElementType, FC } from 'react'
import { memo } from 'react'
import { Streamdown } from 'streamdown'

import { ExternalLink, ExternalLinkIcon } from '@/lib/external-link'
import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { cn } from '@/lib/utils'

// Compact markdown renderer for tool detail bodies. Same Streamdown pipeline
// as the file preview pane, with tighter typography and external-link routing
// so tools that emit markdown (tables, headings, links) render properly
// instead of being dumped as raw text.
//
// LaTeX: wire the same memoized KaTeX math plugin the main chat uses (the raw
// <Streamdown> does NOT enable math by default, so `$x^2$` / `$$…$$` would
// otherwise render as literal text). KaTeX CSS is imported globally in
// styles.css. `singleDollarTextMath` enables inline `$…$`.
const mathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

// Tightened vertical rhythm: model markdown (headings, ---, paragraphs) stacked
// too much whitespace between blocks. Halved the block margins + snugger leading
// so multi-paragraph answers read compact. (We do NOT strip the model's \n\n —
// that would break headings/lists/tables; we just render the gaps smaller.)
const TAG_CLASSES = {
  blockquote: 'mt-1.5 mb-1.5 border-l-2 border-border/70 pl-2.5 italic text-muted-foreground/85',
  h1: 'mt-2 mb-1 text-sm font-semibold tracking-tight text-foreground first:mt-0',
  h2: 'mt-2 mb-1 text-[0.82rem] font-semibold tracking-tight text-foreground first:mt-0',
  h3: 'mt-1.5 mb-0.5 text-[0.78rem] font-semibold text-foreground first:mt-0',
  h4: 'mt-1.5 mb-0.5 text-[0.74rem] font-semibold text-foreground first:mt-0',
  hr: 'my-1.5 border-border/50',
  li: 'marker:text-muted-foreground/60',
  ol: 'mb-1 list-decimal pl-5 last:mb-0',
  p: 'mb-1 leading-snug last:mb-0',
  pre: 'mb-1.5 overflow-x-auto rounded-md border border-border/60 bg-background/70 p-2 font-mono text-[0.7rem] leading-[1.5] last:mb-0',
  td: 'px-2 py-1 align-top leading-snug',
  th: 'px-2 py-1 text-left text-[0.62rem] font-semibold uppercase tracking-[0.08em] text-muted-foreground/80',
  thead: 'bg-muted/40',
  ul: 'mb-1 list-disc pl-5 last:mb-0'
} as const

function tagged<T extends keyof typeof TAG_CLASSES>(Tag: T) {
  const Component = (({ className, ...rest }: ComponentProps<T>) => {
    const Element = Tag as ElementType

    return <Element className={cn(TAG_CLASSES[Tag], className)} {...rest} />
  }) as FC<ComponentProps<T>>

  Component.displayName = `Md.${Tag}`

  return Component
}

function MarkdownAnchor({ children, className, href, ...rest }: ComponentProps<'a'>) {
  if (!href || !/^https?:\/\//i.test(href)) {
    return (
      <a
        className={cn('font-medium underline underline-offset-4 decoration-current/20', className)}
        href={href}
        {...rest}
      >
        {children}
      </a>
    )
  }

  return (
    <ExternalLink className={cn('decoration-current/20', className)} href={href} showExternalIcon={false}>
      {children}
      <ExternalLinkIcon />
    </ExternalLink>
  )
}

function MarkdownCode({ className, ...rest }: ComponentProps<'code'>) {
  return (
    <code
      className={cn('rounded bg-muted/80 px-1 py-px font-mono text-[0.86em] text-muted-foreground', className)}
      {...rest}
    />
  )
}

function MarkdownTable({ className, ...rest }: ComponentProps<'table'>) {
  return (
    <div className="mb-2 max-w-full overflow-x-auto rounded-md border border-border/60 last:mb-0">
      <table
        className={cn(
          'w-full border-collapse text-[0.72rem] [&_tr]:border-b [&_tr]:border-border/50 last:[&_tr]:border-0',
          className
        )}
        {...rest}
      />
    </div>
  )
}

const COMPONENTS = {
  a: MarkdownAnchor,
  blockquote: tagged('blockquote'),
  code: MarkdownCode,
  h1: tagged('h1'),
  h2: tagged('h2'),
  h3: tagged('h3'),
  h4: tagged('h4'),
  hr: tagged('hr'),
  li: tagged('li'),
  ol: tagged('ol'),
  p: tagged('p'),
  pre: tagged('pre'),
  table: MarkdownTable,
  td: tagged('td'),
  th: tagged('th'),
  thead: tagged('thead'),
  ul: tagged('ul')
}

// ★ 性能: memo 包裹, 浅比 className/text/streaming (全是原语)。历史 monitor/watcher/
//   assistant 卡片被 waterfall 的 Row 每 80ms flush 带着重渲时, 若 text/streaming 未变
//   则跳过 Streamdown 解析。与 Row 的 memo 叠加 → 流式期只有活跃 bubble 真正重解析。
export const CompactMarkdown = memo(function CompactMarkdown({
  className,
  text,
  streaming = false
}: {
  className?: string
  text: string
  // While streaming, enable Streamdown's incomplete-markdown handling so a
  // half-written ``**bold``, unclosed code fence, or partial table renders
  // gracefully instead of flashing raw syntax mid-stream.
  streaming?: boolean
}) {
  return (
    <div className={cn('max-w-full text-xs leading-relaxed text-muted-foreground/90 wrap-anywhere', className)}>
      <Streamdown
        components={COMPONENTS}
        controls={false}
        mode="static"
        parseIncompleteMarkdown={streaming}
        plugins={{ math: mathPlugin }}
      >
        {text}
      </Streamdown>
    </div>
  )
})
