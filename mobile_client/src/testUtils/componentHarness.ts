// Function-call component harness for screen behavior tests — the shared version of the
// TE-12 in-file harness RegisterScreen.test.tsx pioneered.
//
// The repo deliberately ships NO renderer (no react-test-renderer, react-dom, or
// @testing-library/react-native) and CI forbids new deps, so screen tests drive the real
// component as a plain function: React's stateful hooks are swapped for an index-keyed cell
// store and every state set synchronously re-invokes the component to produce a fresh element
// tree (mirroring setState → re-render).
//
// Contract for screens tested this way: only useState / useRef / useMemo / useCallback plus
// module-mocked hooks (useNavigation, useFocusEffect, useTraining, useThemeTokens, …). Data
// loading belongs behind useFocusEffect, which tests mock at the @react-navigation/native seam
// and trigger explicitly — useEffect is a no-op here by design.
//
// Usage in a test file (factory must require lazily — jest.mock hoists above imports):
//   jest.mock('react', () => require('../testUtils/componentHarness').createReactMock());
import type * as ReactTypes from 'react';

export type AnyElement = ReactTypes.ReactElement<Record<string, unknown>>;

interface HarnessState {
  cells: unknown[];
  idx: number;
  tree: ReactTypes.ReactNode;
  component: (() => ReactTypes.ReactNode) | null;
}

export const harness: HarnessState = { cells: [], idx: 0, tree: null, component: null };

function rerender(): void {
  if (!harness.component) return;
  harness.idx = 0;
  harness.tree = harness.component();
}

/** The `react` module mock: real React with the stateful hooks swapped for the cell store. */
export function createReactMock(): Record<string, unknown> {
  const actual = jest.requireActual<Record<string, unknown>>('react');
  const useState = <S,>(init: S | (() => S)): [S, (v: S | ((p: S) => S)) => void] => {
    const i = harness.idx++;
    if (i >= harness.cells.length) {
      harness.cells[i] = typeof init === 'function' ? (init as () => S)() : init;
    }
    const set = (v: S | ((p: S) => S)): void => {
      harness.cells[i] =
        typeof v === 'function' ? (v as (p: S) => S)(harness.cells[i] as S) : v;
      rerender();
    };
    return [harness.cells[i] as S, set];
  };
  const useRef = <T,>(init: T): { current: T } => {
    const i = harness.idx++;
    if (i >= harness.cells.length) harness.cells[i] = { current: init };
    return harness.cells[i] as { current: T };
  };
  return {
    ...actual,
    useState,
    useRef,
    useMemo: <T,>(fn: () => T): T => fn(),
    useCallback: <F,>(fn: F): F => fn,
    useEffect: (): void => {},
    useLayoutEffect: (): void => {},
  };
}

/** Mount: reset the cell store and invoke the component function for the first tree. */
export function renderComponent(fn: () => ReactTypes.ReactNode): void {
  harness.cells = [];
  harness.component = fn;
  rerender();
}

/** Depth-first element list under `node` (defaults to the current tree). */
export function elementsIn(node: ReactTypes.ReactNode): AnyElement[] {
  const ReactActual = jest.requireActual<typeof ReactTypes>('react');
  const out: AnyElement[] = [];
  const walk = (n: ReactTypes.ReactNode): void => {
    if (Array.isArray(n)) {
      n.forEach(walk);
      return;
    }
    if (ReactActual.isValidElement(n)) {
      const el = n as AnyElement;
      out.push(el);
      walk(el.props.children as ReactTypes.ReactNode);
    }
  };
  walk(node);
  return out;
}

export function allElements(): AnyElement[] {
  return elementsIn(harness.tree);
}

/** All string content under a node, concatenated. */
export function textOf(node: ReactTypes.ReactNode): string {
  const ReactActual = jest.requireActual<typeof ReactTypes>('react');
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (ReactActual.isValidElement(node)) {
    return textOf((node as AnyElement).props.children as ReactTypes.ReactNode);
  }
  return '';
}

export function screenText(): string {
  return textOf(harness.tree);
}

export function pressables(root?: ReactTypes.ReactNode): AnyElement[] {
  const els = root === undefined ? allElements() : elementsIn(root);
  return els.filter((e) => typeof e.props.onPress === 'function');
}

export function pressableByLabel(label: string, root?: ReactTypes.ReactNode): AnyElement {
  const found = pressables(root).find((e) => e.props.accessibilityLabel === label);
  if (!found) throw new Error(`no pressable labeled "${label}"`);
  return found;
}

export async function press(el: AnyElement): Promise<void> {
  await (el.props.onPress as () => unknown)();
  await flush();
}

/**
 * The first FlatList-like element (has `data` + `renderItem`), rendered to its row trees —
 * virtualized lists never invoke renderItem in a function-call harness, so tests do.
 */
export function flatListRows(root?: ReactTypes.ReactNode): ReactTypes.ReactNode[] {
  const els = root === undefined ? allElements() : elementsIn(root);
  const list = els.find(
    (e) => typeof e.props.renderItem === 'function' && Array.isArray(e.props.data),
  );
  if (!list) throw new Error('no FlatList found in the tree');
  const data = list.props.data as unknown[];
  const renderItem = list.props.renderItem as (info: {
    item: unknown;
    index: number;
  }) => ReactTypes.ReactNode;
  return data.map((item, index) => renderItem({ item, index }));
}

/** Drain pending microtasks/immediates (async state settles before assertions). */
export function flush(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}
