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
