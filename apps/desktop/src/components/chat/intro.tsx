export type IntroProps = {
  personality?: string
  seed?: number
}

// 主 Agent 对话顶部常驻的多模态引导气泡。形态与 web 的 "System" 消息一致 (也就是和
// 本页的 You / Assistant 消息同一套骨架): 头像列 (i) + 头部行 (角色名) + 正文气泡,
// 左对齐、与下方消息左缘对齐, 字号走 --conversation-text-font-size。发送消息后不
// 消失, 始终位于消息列最顶部。
// NOTE: 文案与角色名都写死英文 (与 web 引导气泡一致) —— 未进 i18n 词表。
const MM_WELCOME =
  'Turn on the camera or share your screen, then just ask. One-shot visual questions go to QueryWorker, which reads the frames from the moment you asked and, when needed, recalls history or searches for reference material.'

export function Intro(_props: IntroProps) {
  return (
    <div className="flex w-full min-w-0 flex-row gap-2 pb-(--conversation-turn-gap)" data-slot="aui_intro">
      <div
        aria-hidden="true"
        className="flex size-7 shrink-0 select-none items-center justify-center rounded-full bg-(--ui-bg-tertiary) text-[0.7rem] font-semibold text-(--ui-text-secondary)"
      >
        i
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-0">
        <div className="mb-1 flex items-center gap-1.5 text-[0.65rem] text-(--ui-text-tertiary)">
          <span className="font-medium text-(--ui-text-secondary)">System</span>
        </div>
        <div
          className="wrap-anywhere min-w-0 max-w-full rounded-lg bg-(--ui-bg-elevated) px-3 py-2 text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
          data-selectable-text="true"
        >
          {MM_WELCOME}
        </div>
      </div>
    </div>
  )
}
