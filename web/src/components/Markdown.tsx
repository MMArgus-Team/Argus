import { memo, useMemo, type ReactNode } from "react";
// KaTeX is a hard dependency of web/ (see web/package.json). It MUST be a
// static import — a dynamic ``import(/* @vite-ignore */ "katex")`` with a
// variable specifier stays in the browser bundle verbatim, the browser cannot
// resolve the bare specifier at runtime, and every LaTeX block would degrade
// to raw TeX even though the package ships with the app.
import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * Lightweight markdown renderer for LLM output.
 * Handles: code blocks, inline code, bold, italic, headers, links, lists, horizontal rules.
 * NOT a full CommonMark parser — optimized for typical assistant message patterns.
 *
 * `streaming` renders a blinking caret at the tail of the last block so it
 * appears to hug the final character instead of wrapping onto a new line
 * after a block element (paragraph/list/code/…).
 */
function MarkdownInner({
  content,
  highlightTerms,
  streaming,
}: {
  content: string;
  highlightTerms?: string[];
  streaming?: boolean;
}) {
  // ★ 性能(治大表格流式硬卡死): 流式期间【降级为纯文本】, 不跑 parseBlocks/parseInline。
  //   parseBlocks 每次 content 变(每 80ms flush)都全量重扫整串, 对增长中的大 GFM 表格
  //   是 O(n²)+每格重建, 单 flush 超帧预算 → 连续 flush livelock 卡死。流式期间用户只需
  //   看到文字在长出来, 纯文本 whitespace-pre-wrap 渲染是 O(1) diff; 完成(streaming=false)
  //   后再跑一次完整 Markdown。这样表格/代码/公式最终仍完整渲染, 但生成期不再卡。
  if (streaming) {
    return (
      <div className="text-sm text-foreground leading-snug whitespace-pre-wrap break-words">
        {content}
        <StreamingCaret />
      </div>
    );
  }
  return <ParsedMarkdown content={content} highlightTerms={highlightTerms} />;
}

// Full-parse render path (non-streaming / completed messages).
function ParsedMarkdown({
  content, highlightTerms,
}: { content: string; highlightTerms?: string[] }) {
  const blocks = useMemo(() => parseBlocks(content), [content]);
  // space-y-1 (was space-y-2) + snugger leading: model answers stacked too much
  // whitespace between paragraphs/headings/---. Tightened without touching the
  // model's \n\n (which drives the block split).
  return (
    <div className="text-sm text-foreground leading-snug space-y-1">
      {blocks.map((block, i) => (
        <Block key={i} block={block} highlightTerms={highlightTerms} caret={null} />
      ))}
    </div>
  );
}

/**
 * Memoized wrapper. In the multimodal chat page, ANY setState (e.g. one
 * assistant bubble streaming, or eventLog updating, or ctx panel refreshing)
 * re-renders the parent, and without memo every completed assistant bubble
 * re-parses its whole content each time. `memo` skips that when props are
 * identical. Uses a custom shallow compare that treats `highlightTerms` by
 * reference (callers pass a stable ref or undefined).
 */
export const Markdown = memo(MarkdownInner, (a, b) => (
  a.content === b.content &&
  a.streaming === b.streaming &&
  a.highlightTerms === b.highlightTerms
));

function StreamingCaret() {
  return (
    <span
      aria-hidden
      className="inline-block w-[0.5em] h-[1em] ml-0.5 align-[-0.15em] bg-foreground/50 animate-pulse"
    />
  );
}

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type BlockNode =
  | { type: "code"; lang: string; content: string }
  | { type: "heading"; level: number; content: string }
  | { type: "hr" }
  | { type: "list"; ordered: boolean; items: string[]; start?: number }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "paragraph"; content: string };

/** Split a GFM table row "| a | b |" into cells (trim outer pipes, respect
 *  escaped \| ). Returns trimmed cell strings. */
function _splitTableRow(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  // Split on unescaped pipes.
  const cells: string[] = [];
  let buf = "";
  for (let k = 0; k < s.length; k++) {
    if (s[k] === "\\" && s[k + 1] === "|") { buf += "|"; k++; continue; }
    if (s[k] === "|") { cells.push(buf.trim()); buf = ""; continue; }
    buf += s[k];
  }
  cells.push(buf.trim());
  return cells;
}

/** A line is a GFM table separator if every cell is like ---, :--, --:, :-:. */
function _isTableSeparator(line: string): boolean {
  if (!line.includes("|") && !line.includes("-")) return false;
  const cells = _splitTableRow(line);
  if (cells.length === 0) return false;
  return cells.every((c) => /^:?-{1,}:?$/.test(c.replace(/\s/g, "")));
}

/* ------------------------------------------------------------------ */
/*  Block parser                                                       */
/* ------------------------------------------------------------------ */

function parseBlocks(text: string): BlockNode[] {
  const lines = text.split("\n");
  const blocks: BlockNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    const fenceMatch = line.match(/^```(\w*)/);
    if (fenceMatch) {
      const lang = fenceMatch[1] || "";
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push({ type: "code", lang, content: codeLines.join("\n") });
      continue;
    }

    // Heading
    const headingMatch = line.match(/^(#{1,4})\s+(.+)/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        content: headingMatch[2],
      });
      i++;
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    // Unordered list
    if (/^[-*+]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*+]\s/, ""));
        i++;
      }
      blocks.push({ type: "list", ordered: false, items });
      continue;
    }

    // Ordered list
    const _olm = line.match(/^(\d+)[.)]\s/);
    if (_olm) {
      // ★ 记住这一段有序列表的【起始序号】(原文第一项的真实数字)。lightweight 解析器
      //   会把"1. 标题 + 子项(-) + 1. 标题…"拆成多个单项 <ol>, 若都靠 CSS list-decimal
      //   自动编号, 每段都从 1 开始 → 界面出现一堆 "1."。用 <ol start> 保住真实序号。
      const start = parseInt(_olm[1], 10) || 1;
      const items: string[] = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+[.)]\s/, ""));
        i++;
      }
      blocks.push({ type: "list", ordered: true, items, start });
      continue;
    }

    // GFM table: a header row (contains a pipe) immediately followed by a
    // separator row (|---|:--:|...). Rows continue until a non-pipe/blank line.
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      _isTableSeparator(lines[i + 1])
    ) {
      const header = _splitTableRow(line);
      i += 2; // consume header + separator
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim() !== "" && lines[i].includes("|")) {
        rows.push(_splitTableRow(lines[i]));
        i++;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    // Empty line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph — collect consecutive non-empty, non-special lines
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].match(/^```/) &&
      !lines[i].match(/^#{1,4}\s/) &&
      !lines[i].match(/^[-*+]\s/) &&
      !lines[i].match(/^\d+[.)]\s/) &&
      !lines[i].match(/^[-*_]{3,}\s*$/) &&
      // stop if a GFM table starts here (pipe header + separator next line)
      !(lines[i].includes("|") && i + 1 < lines.length && _isTableSeparator(lines[i + 1]))
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", content: paraLines.join("\n") });
    }
  }

  return blocks;
}

/* ------------------------------------------------------------------ */
/*  Block renderer                                                     */
/* ------------------------------------------------------------------ */

function BlockInner({
  block,
  highlightTerms,
  caret,
}: {
  block: BlockNode;
  highlightTerms?: string[];
  caret?: ReactNode;
}) {
  switch (block.type) {
    case "code":
      return (
        <pre className="bg-secondary/60 border border-border px-3 py-2.5 text-xs font-mono leading-relaxed overflow-x-auto">
          <code>
            {block.content}
            {caret}
          </code>
        </pre>
      );

    case "heading": {
      const Tag = `h${Math.min(block.level, 4)}` as "h1" | "h2" | "h3" | "h4";
      const sizes: Record<string, string> = {
        h1: "text-base font-bold",
        h2: "text-sm font-bold",
        h3: "text-sm font-semibold",
        h4: "text-sm font-medium",
      };
      return (
        <Tag className={sizes[Tag]}>
          <InlineContent text={block.content} highlightTerms={highlightTerms} />
          {caret}
        </Tag>
      );
    }

    case "hr":
      return (
        <>
          <hr className="border-border" />
          {caret}
        </>
      );

    case "list": {
      const Tag = block.ordered ? "ol" : "ul";
      const last = block.items.length - 1;
      return (
        <Tag
          // ★ start: 保住有序列表的真实起始序号 (拆成多个 <ol> 时才不会都从 1 开始)。
          {...(block.ordered && block.start && block.start !== 1
            ? { start: block.start } : {})}
          className={`space-y-0.5 ${block.ordered ? "list-decimal" : "list-disc"} pl-5 text-sm`}
        >
          {block.items.map((item, i) => (
            <li key={i}>
              <InlineContent text={item} highlightTerms={highlightTerms} />
              {i === last ? caret : null}
            </li>
          ))}
        </Tag>
      );
    }

    case "table":
      return (
        <div className="overflow-x-auto">
          <table className="my-1 w-full border-collapse text-xs">
            <thead>
              <tr>
                {block.header.map((h, i) => (
                  <th
                    key={i}
                    className="border border-border bg-secondary/50 px-2 py-1 text-left font-semibold"
                  >
                    <InlineContent text={h} highlightTerms={highlightTerms} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr key={r}>
                  {block.header.map((_, c) => (
                    <td key={c} className="border border-border px-2 py-1 align-top">
                      <InlineContent text={row[c] ?? ""} highlightTerms={highlightTerms} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {caret}
        </div>
      );

    case "paragraph":
      return (
        <p>
          <InlineContent text={block.content} highlightTerms={highlightTerms} />
          {caret}
        </p>
      );
  }
}

/** Cheap structural equality for two parsed blocks. parseBlocks 每次流式重解析都会
 *  产出【全新的 block 对象】(引用必变), 所以裸 React.memo 命中不了 —— 必须按值比较。
 *  流式追加时只有【最后一个 block】在长, 前面所有 block 值不变 → memo 命中、跳过重渲,
 *  否则一份 4 表 600 格的报告每 80ms flush 全量 reconcile 600+ 节点, 叠加面板输出直接
 *  撑爆主线程 (livelock)。 */
function _blockEq(a: BlockNode, b: BlockNode): boolean {
  if (a.type !== b.type) return false;
  switch (a.type) {
    case "code":
      return a.content === (b as typeof a).content && a.lang === (b as typeof a).lang;
    case "heading":
      return a.level === (b as typeof a).level && a.content === (b as typeof a).content;
    case "hr":
      return true;
    case "list": {
      const bb = b as typeof a;
      return a.ordered === bb.ordered
        && a.start === bb.start
        && a.items.length === bb.items.length
        && a.items.every((it, i) => it === bb.items[i]);
    }
    case "table": {
      const bb = b as typeof a;
      if (a.header.length !== bb.header.length
          || !a.header.every((h, i) => h === bb.header[i])) return false;
      if (a.rows.length !== bb.rows.length) return false;
      return a.rows.every((row, r) =>
        row.length === bb.rows[r].length
        && row.every((c, i) => c === bb.rows[r][i]));
    }
    case "paragraph":
      return a.content === (b as typeof a).content;
    default:
      return false;
  }
}

const Block = memo(BlockInner, (a, b) =>
  // caret 只挂在最后一个 block: caret 变化 (从有到无/无到有) 必须重渲。
  (!!a.caret === !!b.caret)
  && a.highlightTerms === b.highlightTerms
  && _blockEq(a.block, b.block),
);

/* ------------------------------------------------------------------ */
/*  Inline parser + renderer                                           */
/* ------------------------------------------------------------------ */

type InlineNode =
  | { type: "text"; content: string }
  | { type: "code"; content: string }
  | { type: "bold"; content: string }
  | { type: "italic"; content: string }
  | { type: "link"; text: string; href: string }
  | { type: "math"; content: string; display: boolean }
  | { type: "br" };

function parseInline(text: string): InlineNode[] {
  const nodes: InlineNode[] = [];
  // Pattern priority: math > code > link > bold > italic > bare URL > line break.
  // Math forms (LaTeX): $$...$$ / \[...\] (display), $...$ / \(...\) (inline).
  // Display forms come first so $$ isn't split by the single-$ rule.
  const pattern =
    /(\$\$([\s\S]+?)\$\$)|(\\\[([\s\S]+?)\\\])|(\$(?!\s)([^$\n]+?)(?<!\s)\$)|(\\\(([\s\S]+?)\\\))|(`[^`]+`)|(\[([^\]]+)\]\(([^)]+)\))|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(\bhttps?:\/\/[^\s<>)\]]+)|(\n)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push({ type: "text", content: text.slice(lastIndex, match.index) });
    }

    if (match[1]) {
      // $$display math$$
      nodes.push({ type: "math", content: match[2], display: true });
    } else if (match[3]) {
      // \[display math\]
      nodes.push({ type: "math", content: match[4], display: true });
    } else if (match[5]) {
      // $inline math$
      nodes.push({ type: "math", content: match[6], display: false });
    } else if (match[7]) {
      // \(inline math\)
      nodes.push({ type: "math", content: match[8], display: false });
    } else if (match[9]) {
      // Inline code
      nodes.push({ type: "code", content: match[9].slice(1, -1) });
    } else if (match[10]) {
      // [text](url) link
      nodes.push({ type: "link", text: match[11], href: match[12] });
    } else if (match[13]) {
      // **bold**
      nodes.push({ type: "bold", content: match[14] });
    } else if (match[15]) {
      // *italic*
      nodes.push({ type: "italic", content: match[16] });
    } else if (match[17]) {
      // Bare URL
      nodes.push({ type: "link", text: match[17], href: match[17] });
    } else if (match[18]) {
      // Line break within paragraph
      nodes.push({ type: "br" });
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    nodes.push({ type: "text", content: text.slice(lastIndex) });
  }

  return nodes;
}

const InlineContent = memo(function InlineContent({
  text,
  highlightTerms,
}: {
  text: string;
  highlightTerms?: string[];
}) {
  const nodes = useMemo(() => parseInline(text), [text]);

  return (
    <>
      {nodes.map((node, i) => {
        switch (node.type) {
          case "text":
            return (
              <HighlightedText
                key={i}
                text={node.content}
                terms={highlightTerms}
              />
            );
          case "code":
            return (
              <code
                key={i}
                className="bg-secondary/60 px-1.5 py-0.5 text-xs font-mono text-primary/90"
              >
                {node.content}
              </code>
            );
          case "bold":
            return (
              <strong key={i} className="font-semibold">
                <HighlightedText text={node.content} terms={highlightTerms} />
              </strong>
            );
          case "italic":
            return (
              <em key={i}>
                <HighlightedText text={node.content} terms={highlightTerms} />
              </em>
            );
          case "link": {
            // Security: only render http(s)/mailto links. Other schemes
            // (javascript:, data:, vbscript:) are dropped to plain text so a
            // crafted link in agent/message content can't execute on click.
            const href = node.href.trim();
            if (!/^(https?:|mailto:)/i.test(href)) {
              return (
                <HighlightedText
                  key={i}
                  text={node.text}
                  terms={highlightTerms}
                />
              );
            }
            return (
              <a
                key={i}
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-primary underline underline-offset-2 decoration-primary/30 hover:decoration-primary/60 transition-colors"
              >
                {node.text}
              </a>
            );
          }
          case "math":
            return <KatexMath key={i} tex={node.content} display={node.display} />;
          case "br":
            return <br key={i} />;
        }
      })}
    </>
  );
});

/**
 * Render a LaTeX snippet with KaTeX. KaTeX is statically imported (hard
 * dependency), so rendering is synchronous; a render error degrades to the
 * raw TeX in a mono span rather than crash or show nothing.
 */
const KatexMath = memo(function KatexMath({
  tex,
  display,
}: {
  tex: string;
  display: boolean;
}) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        output: "html",
      });
    } catch {
      return null;
    }
  }, [tex, display]);

  // Fallback: render error → show raw TeX (never blank).
  if (html === null) {
    return (
      <code className={display
        ? "block my-1 bg-secondary/40 px-2 py-1 text-xs font-mono"
        : "px-1 text-xs font-mono text-primary/90"}>
        {display ? tex : `$${tex}$`}
      </code>
    );
  }
  return display ? (
    <span className="block my-1 overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
});

/** Highlight search terms within a plain text string. */
function HighlightedText({ text, terms }: { text: string; terms?: string[] }) {
  // ★ 过滤空/纯空白 term: 空字符串会让 (a||b) 这类交替匹配到【每个位置的空串】,
  //   既产生错误高亮又是潜在的退化匹配源。空 term 列表 → 直接原文返回。
  const clean = (terms ?? []).filter((t) => t && t.trim() !== "");
  if (clean.length === 0) return <>{text}</>;

  // Build a regex that matches any of the search terms (case-insensitive)
  const escaped = clean.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  // split-capture regex (仅用于切分, 每次都新建避免共享 lastIndex 状态)。
  const splitRe = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(splitRe);
  // ★ 判定"这一段是否命中" 不能复用带 /g 的正则做 .test() —— /g 正则是【有状态】的,
  //   连续 .test() 会推进 lastIndex, 造成隔一个漏一个的错误高亮。改成用一个不带 /g
  //   的整段匹配正则 (^(...)$) 逐段判定, 无状态、结果确定。
  const isTermRe = new RegExp(`^(?:${escaped.join("|")})$`, "i");

  return (
    <>
      {parts.map((part, i) =>
        part && isTermRe.test(part) ? (
          <mark key={i} className="bg-warning/30 text-warning px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}
