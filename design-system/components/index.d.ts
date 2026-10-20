import type * as React from 'react';

/** Kadmium components. Loaded as `components/bundle.js` (one classic script) they attach to `window.Kadmium`; they expect React 18 as `window.React` / `window.ReactDOM` and `tokens.css` + `components/bundle.css` on the page. */

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** `primary` is cadmium — at most one per view, for the thing the view is for. `cloud` is cherenkov — only for a Kadmium Cloud action (attach a burst GPU, upgrade, open the managed plane). `quiet` is the default. `ghost` for toolbars and card headers. `danger` for destructive actions, always behind a confirmation. */
  variant?: 'primary' | 'cloud' | 'quiet' | 'ghost' | 'danger';
  /** `md` (36px) by default; `sm` (28px) inside tables and card headers; `lg` (44px) for hero and plan-card calls to action. */
  size?: 'sm' | 'md' | 'lg';
  /** A 16px leading icon: the name of a built-in glyph (`rods`, `cloud`, `gpu`, `desktop`, `laptop`, `compact`, `sbc`, `nas`, `check`, `copy`) or your own SVG element. */
  icon?: 'desktop' | 'laptop' | 'compact' | 'sbc' | 'nas' | 'gpu' | 'cloud' | 'rods' | 'check' | 'copy' | React.ReactNode;
}
export declare function Button(props: ButtonProps): React.ReactElement;

export interface BadgeProps {
  /** `local` (cadmium) and `cloud` (cherenkov) say where compute runs; `ok` / `warn` / `danger` say how it is doing; `neutral` for everything else (model names, protocols). */
  tone?: 'neutral' | 'local' | 'cloud' | 'ok' | 'warn' | 'danger';
  /** A 6px status dot before the word. Use it for live states (resident, yielded, offline). */
  dot?: boolean;
  title?: string;
  className?: string;
  /** One or two words, set in mono uppercase by the component. */
  children?: React.ReactNode;
}
export declare function Badge(props: BadgeProps): React.ReactElement;

export interface MeterProps {
  /** What is measured — `VRAM`, `UMA`, `RAM`, `NVMe`, `link`. Rendered uppercase in mono. */
  label: string;
  value: number;
  max: number;
  /** Appended to the value text, e.g. `GB`, `Gbps`. */
  unit?: string;
  /** `local` fills cadmium; `cloud` fills cherenkov. A NodeCard sets this from its placement. */
  tone?: 'local' | 'cloud';
  /** Force the danger fill (e.g. a node reporting thermal throttling). */
  critical?: boolean;
  /** Fraction (0–1) at and above which the fill turns danger, e.g. `0.95` for VRAM. */
  criticalAt?: number;
  /** Replace the default `value / max unit` text. */
  format?: (value: number, max: number) => string;
  className?: string;
}
export declare function Meter(props: MeterProps): React.ReactElement;

export interface NodeCardProps {
  /** The node's name as the person named it: `M4 Mac Mini`, `Gaming laptop (Ava)`, `RunPod A100`. */
  name: string;
  /** One mono line of hardware: `rtx 5070 · 12 gb · nvme 7 gb/s`. Lowercase, dot-separated. */
  hardware?: string;
  /** Local hardware or rented/managed cloud. Drives the badge, the icon tint and the default meter colour. */
  placement?: 'local' | 'cloud';
  /** Which glyph to show. Defaults to `desktop` for local and `cloud` for cloud nodes. */
  kind?: 'desktop' | 'laptop' | 'compact' | 'sbc' | 'nas' | 'gpu' | 'cloud';
  /** Live state → status badge: resident/training (ok), prefetching (cloud), joining (neutral), yielded/throttled (warn), offline/faulted (danger). */
  status?: 'resident' | 'training' | 'prefetching' | 'joining' | 'yielded' | 'throttled' | 'offline' | 'faulted';
  /** The shard range this node holds, in mono: `layers 12–31`, `experts 40–63`. */
  shards?: string;
  /** Memory-tier meters, stacked. */
  meters?: MeterProps[];
  /** Anything else to show between the meters and the footer. */
  children?: React.ReactNode;
  className?: string;
}
export declare function NodeCard(props: NodeCardProps): React.ReactElement;

export interface ShardRailNode {
  id?: string;
  /** Short label under the rod (mono, uppercase): `mac`, `pc`, `lap-1`, `pi`, `a100`. */
  name: string;
  /** This node's share of the plan, 0–1; the rod's depth is proportional. 0 draws an idle (graphite) rod. */
  share: number;
  placement?: 'local' | 'cloud';
}
export interface ShardRailProps {
  nodes: ShardRailNode[];
  /** Horizontal distance between rods in px (default 56). Keep it a multiple of 8. */
  pitch?: number;
  /** Depth of a 100 % rod in px (default 96). */
  depth?: number;
  /** Hide the percentage under each label. */
  showShare?: boolean;
  /** Accessible title for the drawing, e.g. `Shard plan for deepseek-v3-671b`. */
  label?: string;
  className?: string;
}
export declare function ShardRail(props: ShardRailProps): React.ReactElement;

export interface PlanCardProps {
  /** `self-hosted` (cadmium eyebrow, primary button) or `cloud` (cherenkov eyebrow, cloud button). */
  product: 'self-hosted' | 'cloud';
  /** Override the eyebrow text (`Self-hosted` / `Kadmium Cloud` by default). */
  eyebrow?: string;
  name: string;
  /** The price as text, e.g. `$0`, `$0.89`, `Custom`. */
  price: string;
  /** e.g. `/ GPU-hour`, `forever`. */
  period?: string;
  summary?: string;
  features?: string[];
  /** Button label. Omit for a card without a call to action. */
  cta?: string;
  onSelect?: () => void;
  className?: string;
}
export declare function PlanCard(props: PlanCardProps): React.ReactElement;

export interface CodeBlockProps {
  code: string;
  /** Drives the small highlighter: prompts and flags (shell), keywords and macros (rust), keys and values (toml, json). `text` highlights nothing. */
  language?: 'shell' | 'rust' | 'toml' | 'json' | 'text';
  /** Shown in the header, e.g. a filename or `terminal`. Defaults to the language name. */
  title?: string;
  /** Show the Copy button (default true). */
  copy?: boolean;
  className?: string;
}
export declare function CodeBlock(props: CodeBlockProps): React.ReactElement;

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  /** Replaces the hint and marks the field invalid. */
  error?: string;
  /** Mono text — hostnames, ports, paths, model ids. */
  mono?: boolean;
  /** A mono prefix inside the field, e.g. `https://`, `$`. */
  prefix?: React.ReactNode;
  className?: string;
}
export declare function Input(props: InputProps): React.ReactElement;

export interface TabItem { id: string; label: string; count?: number }
export interface TabsProps {
  items: TabItem[];
  /** The selected item's id. Controlled: pair it with onChange. */
  value: string;
  onChange?: (id: string) => void;
  /** Accessible name of the tab list. */
  label?: string;
  className?: string;
}
export declare function Tabs(props: TabsProps): React.ReactElement;

export interface MarkProps {
  /** Rendered size in px (default 24). */
  size?: number;
  /** `brand`: cadmium rods, rail in currentColor. `mono`: everything in currentColor. */
  tone?: 'brand' | 'mono';
  title?: string;
  className?: string;
}
export declare function Mark(props: MarkProps): React.ReactElement;

export interface IconProps {
  /** One of the twelve glyphs in assets/Icons: `rod`, `rods`, `desktop`, `laptop`, `compact`, `sbc`, `nas`, `gpu`, `cloud`, `memory`, `check`, `copy`. Drawn in currentColor at 1.75px stroke. */
  name: 'rod' | 'rods' | 'desktop' | 'laptop' | 'compact' | 'sbc' | 'nas' | 'gpu' | 'cloud' | 'memory' | 'check' | 'alert' | 'copy';
  /** Rendered size in px (default 24 — the size the set is drawn at; 16 in buttons, 20 in inputs and node cards). */
  size?: number;
  /** An accessible name. Omit for a purely decorative icon beside its text. */
  title?: string;
  className?: string;
}
export declare function Icon(props: IconProps): React.ReactElement;

declare global {
  interface Window {
    Kadmium: {
      Button: typeof Button; Badge: typeof Badge; Meter: typeof Meter; NodeCard: typeof NodeCard; ShardRail: typeof ShardRail;
      PlanCard: typeof PlanCard; CodeBlock: typeof CodeBlock; Input: typeof Input; Tabs: typeof Tabs; Mark: typeof Mark; Icon: typeof Icon;
    };
  }
}
