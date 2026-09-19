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
