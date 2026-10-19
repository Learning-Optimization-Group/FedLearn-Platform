import { NativeModules, Platform } from 'react-native';
import { foregroundService } from '../lib/foregroundService';

jest.mock('react-native', () => ({
  Platform: { OS: 'android' },
  NativeModules: { FlService: { delay: jest.fn(() => Promise.resolve()) } },
}));

describe('training waits during activity suspension', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    (Platform as { OS: string }).OS = 'android';
    NativeModules.FlService.delay = jest.fn(() => Promise.resolve());
  });
  afterEach(() => { jest.useRealTimers(); });

  it('completes an Android pacing wait without advancing paused JavaScript timers', async () => {
    let complete = false;
    void foregroundService.delay(1500).then(() => { complete = true; });
    await Promise.resolve();
    await Promise.resolve();
    expect(complete).toBe(true);
    expect(jest.getTimerCount()).toBe(0);
    expect(NativeModules.FlService.delay).toHaveBeenCalledWith(1500);
  });

  it('refuses an Android runtime missing native waits rather than hanging on paused timers', async () => {
    NativeModules.FlService.delay = undefined;
    await expect(foregroundService.delay(1500)).rejects.toThrow('Update the Android app');
    expect(jest.getTimerCount()).toBe(0);
  });

  it('refuses missing Android scheduling support before training can start', () => {
    NativeModules.FlService.delay = undefined;
    expect(() => foregroundService.assertSchedulingAvailable()).toThrow('Update the Android app');
    expect(jest.getTimerCount()).toBe(0);
  });

  it('keeps foreground JavaScript timing on iOS', async () => {
    (Platform as { OS: string }).OS = 'ios';
    let complete = false;
    void foregroundService.delay(1500).then(() => { complete = true; });
    await jest.advanceTimersByTimeAsync(1499);
    expect(complete).toBe(false);
    await jest.advanceTimersByTimeAsync(1);
    expect(complete).toBe(true);
    expect(NativeModules.FlService.delay).not.toHaveBeenCalled();
  });

  it.each([-1, NaN, Infinity, .5, 60001])('rejects invalid native wait budget %s', async ms => {
    await expect(foregroundService.delay(ms)).rejects.toThrow('Invalid training delay');
    expect(NativeModules.FlService.delay).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });
});
