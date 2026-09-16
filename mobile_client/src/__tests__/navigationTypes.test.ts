// Pins the stage-1 IA contract without rendering a navigator (no renderer dep exists in this
// repo): 4 tabs — Home (default, listed first), Projects, Models, Settings — plus the three
// registered stack pushes over the tabs. The constants are tied to the ParamLists via
// `satisfies` (navigation/types.ts), and AppNavigator's Screen names are checked against the
// same ParamLists by the typed navigators — tsc --noEmit is CI-gated, so a rename breaks
// compile before it breaks this test.
import { APP_STACK_ROUTES, MAIN_TAB_ROUTES } from '../navigation/types';
import type { MainTabParamList } from '../navigation/types';

describe('main tab bar (4 tabs, was 6)', () => {
  it('has exactly Home, Projects, Models, Settings', () => {
    expect(MAIN_TAB_ROUTES).toEqual(['Home', 'Projects', 'Models', 'Settings']);
  });

  it('lands on Home by default (first tab)', () => {
    expect(MAIN_TAB_ROUTES[0]).toBe('Home');
  });

  it('Home takes no params — the old Projects→Home selection handoff is gone', () => {
    // Compile-time pin (tsc is CI-gated): Home's param type must remain exactly `undefined` —
    // reintroducing a params object flips Exact<> to false and breaks this assignment.
    type Exact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;
    const homeParamsAreExactlyUndefined: Exact<MainTabParamList['Home'], undefined> = true;
    expect(homeParamsAreExactlyUndefined).toBe(true);
  });

  it('no longer exposes the retired tabs', () => {
    for (const retired of ['Training', 'Playground', 'Library', 'Testing']) {
      expect(MAIN_TAB_ROUTES).not.toContain(retired);
    }
  });
});

describe('authenticated app stack', () => {
  it('registers the tabs plus the three stage-2 pushes', () => {
    expect(APP_STACK_ROUTES).toEqual(['MainTabs', 'ModelTesting', 'Playground', 'ProjectDetail']);
  });
});
