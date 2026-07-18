// Typed navigation contracts for the whole app (stage 1 of the IA redesign).
//
// Every navigator is generic over one of these ParamLists so route renames and param-shape
// mistakes fail `tsc --noEmit` (CI-gated) instead of only at runtime — the old
// `useNavigation<any>()` pattern let `navigate('Training', …)` survive the Training→Home
// rename silently. Keep this file free of component imports so tests can assert the route
// tables without pulling screens (and their native deps) into the jest module graph.
import type { NavigatorScreenParams, CompositeScreenProps } from '@react-navigation/native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

/** Root auth gate: Login/Register when signed out, the App shell when signed in. */
export type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  App: NavigatorScreenParams<AppStackParamList> | undefined;
};

/**
 * Authenticated shell: the 4-tab bar plus screens that push OVER the tabs.
 * The pushes render the native-stack header (standard title + back affordance) — they must
 * not assume a tab bar below them.
 */
export type AppStackParamList = {
  MainTabs: NavigatorScreenParams<MainTabParamList> | undefined;
  /** Optional saved-model selection from the Models hub; absent → the newest saved model. */
  ModelTesting: { modelPath?: string } | undefined;
  Playground: undefined;
  ProjectDetail: { projectId: string };
};

/** The 4 tabs (was 6): Home is the default landing tab. */
export type MainTabParamList = {
  /** Optional selection handed over from Projects ("train this project here"). */
  Home: { projectId?: string; projectName?: string } | undefined;
  Projects: undefined;
  Models: undefined;
  Settings: undefined;
};

/** Screen-prop helper for tab screens that also navigate the surrounding app stack. */
export type MainTabScreenProps<T extends keyof MainTabParamList> = CompositeScreenProps<
  BottomTabScreenProps<MainTabParamList, T>,
  NativeStackScreenProps<AppStackParamList>
>;

export type AppStackScreenProps<T extends keyof AppStackParamList> = NativeStackScreenProps<
  AppStackParamList,
  T
>;

export type RootStackScreenProps<T extends keyof RootStackParamList> = NativeStackScreenProps<
  RootStackParamList,
  T
>;

// Route tables as runtime constants so tests can pin the IA (4 tabs, Home first/default,
// the three stack pushes) without rendering a navigator (no renderer dep exists in this repo).
// `satisfies` ties each entry to its ParamList, so a route rename breaks compile here too.
export const MAIN_TAB_ROUTES = [
  'Home',
  'Projects',
  'Models',
  'Settings',
] as const satisfies readonly (keyof MainTabParamList)[];
export const APP_STACK_ROUTES = [
  'MainTabs',
  'ModelTesting',
  'Playground',
  'ProjectDetail',
] as const satisfies readonly (keyof AppStackParamList)[];

// Untyped useNavigation()/useRoute() calls resolve against the root list by default.
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace ReactNavigation {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface RootParamList extends RootStackParamList {}
  }
}
