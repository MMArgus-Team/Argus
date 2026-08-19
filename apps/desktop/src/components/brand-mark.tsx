import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// MM-Argus brand badge. The source asset is dark-mode friendly and includes
// its own circular frame, so it does not need a white backing tile.
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn(
        'inline-flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-full bg-[#06101d] shadow-sm ring-1 ring-cyan-400/20',
        className
      )}
      {...props}
    >
      <img alt="" className="size-full object-cover" src={assetPath('mmargus-logo.webp')} />
    </span>
  )
}
