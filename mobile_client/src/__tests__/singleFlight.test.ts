import { SingleFlight } from '../lib/singleFlight';

test('two start requests launch one training operation', async () => {
  const flight = new SingleFlight();
  let finish!: () => void;
  const task = jest.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
  const first = flight.run(task);
  await flight.run(task);
  expect(task).toHaveBeenCalledTimes(1);
  finish();
  await first;
});

test('a failed attempt releases the gate for a later explicit retry', async () => {
  const flight = new SingleFlight();
  await expect(flight.run(async () => { throw new Error('failed'); })).rejects.toThrow('failed');
  const next = jest.fn().mockResolvedValue(undefined);
  await flight.run(next);
  expect(next).toHaveBeenCalledTimes(1);
});

test('a pending join excludes training and reports the rejected action', async () => {
  const flight = new SingleFlight();
  let finishJoin!: () => void;
  const join = flight.run(() => new Promise<void>((resolve) => { finishJoin = resolve; }));
  const training = jest.fn().mockResolvedValue(undefined);
  const busy = jest.fn();

  await flight.run(training, busy);
  expect(training).not.toHaveBeenCalled();
  expect(busy).toHaveBeenCalledTimes(1);
  finishJoin();
  await join;
  await flight.run(training, busy);
  expect(training).toHaveBeenCalledTimes(1);
});

test('Stop does not report completion until the current native operation releases participation', async () => {
  const flight = new SingleFlight();
  let finishRound!: () => void;
  const running = flight.run(() => new Promise<void>((resolve) => { finishRound = resolve; }));
  const abort = jest.fn().mockResolvedValue(undefined);
  const completed = jest.fn();
  const stopping = flight.abortAndWait(abort).then(completed);
  await Promise.resolve();
  await Promise.resolve();
  expect(abort).toHaveBeenCalledTimes(1);
  expect(completed).not.toHaveBeenCalled();
  finishRound();
  await Promise.all([running, stopping]);
  expect(completed).toHaveBeenCalledTimes(1);
  const rejoin = jest.fn().mockResolvedValue(undefined);
  await flight.run(rejoin);
  expect(rejoin).toHaveBeenCalledTimes(1);
});

test('an abort error still waits for the operation to unwind before it propagates', async () => {
  const flight = new SingleFlight();
  let finish!: () => void;
  const work = flight.run(() => new Promise<void>((resolve) => { finish = resolve; }));
  const failure = new Error('abort unavailable');
  let rejected = false;
  const stopped = flight.abortAndWait(async () => { throw failure; }).catch((e) => {
    expect(e).toBe(failure);
    rejected = true;
  });
  await Promise.resolve();
  await Promise.resolve();
  expect(rejected).toBe(false);
  finish();
  await Promise.all([work, stopped]);
  expect(rejected).toBe(true);
});

test('a registration created while Stop waits is also shut down before completion', async () => {
  const flight = new SingleFlight();
  let finish!: () => void;
  let heartbeatActive = false;
  const joining = flight.run(async () => {
    await new Promise<void>(resolve => { finish = resolve; });
    heartbeatActive = true;
  });
  const abort = jest.fn(async () => { heartbeatActive = false; });
  const stopping = flight.abortAndWait(abort);
  await Promise.resolve();
  finish();
  await Promise.all([joining, stopping]);
  expect(heartbeatActive).toBe(false);
  expect(abort).toHaveBeenCalledTimes(2);
});

test('joining stays excluded while final Stop cleanup is still pending', async () => {
  const flight = new SingleFlight();
  let finishWork!: () => void;
  let finishCleanup!: () => void;
  const work = flight.run(() => new Promise<void>(resolve => { finishWork = resolve; }));
  let calls = 0;
  const abort = jest.fn(() => ++calls === 1 ? Promise.resolve()
    : new Promise<void>(resolve => { finishCleanup = resolve; }));
  const stopping = flight.abortAndWait(abort);
  finishWork();
  await work;
  for (let i = 0; i < 4; i++) await Promise.resolve();
  expect(abort).toHaveBeenCalledTimes(2);
  const join = jest.fn().mockResolvedValue(undefined);
  const busy = jest.fn();
  await flight.run(join, busy);
  expect(join).not.toHaveBeenCalled();
  expect(busy).toHaveBeenCalledTimes(1);
  finishCleanup();
  await stopping;
  await flight.run(join, busy);
  expect(join).toHaveBeenCalledTimes(1);
});
