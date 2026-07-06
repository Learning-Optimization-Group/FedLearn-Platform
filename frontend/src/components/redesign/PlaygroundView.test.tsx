import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { PlaygroundView } from './PlaygroundView';
import * as api from '../../services/apiServices';

vi.mock('../../services/apiServices');
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));

/** Minimal AxiosResponse wrapper — the view only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

const MODEL: api.InferableModel = {
  projectId: 'p1',
  name: 'Story model',
  modelType: 'LLM_LORA',
  modelName: 'tiny-gpt',
  status: 'COMPLETED',
  inputKind: 'generation',
  classes: [],
  supported: true,
};

// FE-3: the WS token stream is only a live preview; the REST response carries
// the authoritative generation. When the broker is unreachable and nothing
// streams, the committed bubble must fall back to the REST body — not end up
// blank.
describe('PlaygroundView — generation falls back to the REST body (FE-3)', () => {
  beforeEach(() => {
    // Broker unreachable: activate() fails the socket instead of connecting,
    // so the /topic/inference subscription never goes live and no token
    // arrives.
    vi.mocked(StompClient).mockImplementation(() => {
      const client = {
        onConnect: null,
        onStompError: null,
        onWebSocketError: null as ((e: unknown) => void) | null,
        active: false,
        subscribe: vi.fn(),
        deactivate: vi.fn(),
        activate: vi.fn(() => {
          client.onWebSocketError?.(new Error('connection refused'));
        }),
      };
      return client as unknown as InstanceType<typeof StompClient>;
    });
    vi.mocked(api.fetchInferableModels).mockResolvedValue(resp([MODEL]));
  });

  it('commits the REST generation text when nothing streamed', async () => {
    vi.mocked(api.runGeneration).mockResolvedValue(
      resp({
        modelType: 'LLM_LORA',
        prompt: 'Tell me a story',
        generatedText: 'Once upon a time the models federated.',
        tokenCount: 8,
        finishReason: 'completed',
      }),
    );

    render(<PlaygroundView />);
    const promptBox = await screen.findByPlaceholderText(/message the model/i);
    fireEvent.change(promptBox, { target: { value: 'Tell me a story' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));

    // The assistant bubble must carry the REST body — a dead stream must not
    // leave a blank bubble.
    expect(await screen.findByText('Once upon a time the models federated.')).toBeInTheDocument();
    expect(api.runGeneration).toHaveBeenCalledWith(
      'p1',
      expect.objectContaining({ prompt: 'Tell me a story' }),
    );
    // The in-flight placeholder is gone once the reply is committed.
    expect(screen.queryByText('Generating…')).not.toBeInTheDocument();
  });
});
