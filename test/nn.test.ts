import { devices, grad, init, nn, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  suite("jax.nn.relu()", () => {
    test("should compute ReLU", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.relu(x);
      expect(y.js()).toEqual([0, 0, 1, 2]);
    });

    test("should compute ReLU gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.relu(x).sum());
      const gx = gradFn(x);
      expect(gx.js()).toEqual([0, 0, 1, 1]);
    });
  });

  suite("jax.nn.sigmoid()", () => {
    test("should compute sigmoid", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.sigmoid(x);
      expect(y).toBeAllclose([0.26894142, 0.5, 0.73105858, 0.88079708]);
    });

    test("should compute sigmoid gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.sigmoid(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0.0000454, 0.19661193, 0.19661193, 0.10499359]);
    });
  });

  suite("jax.nn.softSign()", () => {
    test("should compute softsign", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.softSign(x);
      expect(y).toBeAllclose([-0.5, 0, 0.5, 2 / 3]);
    });

    test("should compute softsign gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.softSign(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([1 / 121, 1 / 4, 1 / 4, 1 / 9]);
    });
  });

  suite("jax.nn.softmax()", () => {
    test("should compute softmax", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = nn.softmax(x);
      expect(y).toBeAllclose([
        [0.09003057, 0.24472848, 0.66524094],
        [0.09003057, 0.24472848, 0.66524094],
      ]);
    });

    test("should compute softmax over 2 axes", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = nn.softmax(x, [0, 1]);
      expect(y).toBeAllclose([
        [0.00426978, 0.01160646, 0.03154963],
        [0.08576079, 0.23312202, 0.6336913],
      ]);
    });

    test("should work with no axes", () => {
      expect(nn.softmax(np.zeros([])).js()).toEqual(1);
      expect(nn.softmax(np.array([1, 2, 3]), []).js()).toEqual([1, 1, 1]);
      expect(nn.softmax(np.zeros([0])).js()).toEqual([]);
    });

    test("sum should be constant", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.softmax(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0, 0, 0, 0]);
    });

    test("should compute softmax gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) =>
        nn
          .softmax(x)
          .mul(np.array([0, 0, 1, 0])) // Select one element
          .sum(),
      );
      const gx = gradFn(x);
      expect(gx).toBeAllclose([
        -1.1246564e-6, -9.1131842e-3, 1.9215752e-1, -1.8304321e-1,
      ]);
    });

    test("is consistent with logSoftmax", () => {
      const x = np.array([-10, -1, 1, 2]);
      const softmax = nn.softmax(x.ref);
      const logSoftmax = nn.logSoftmax(x);
      expect(np.log(softmax)).toBeAllclose(logSoftmax);
    });
  });

  suite("jax.nn.logsumexp()", () => {
    test("computes logsumexp correctly", () => {
      const x = np.array([-10, -1, 1, 2]);
      const y = nn.logsumexp(x);
      expect(y.js()).toBeCloseTo(2.3490167);

      const z = nn.logsumexp(
        np.array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]),
      );
      expect(z.js()).toBeCloseTo(8.45834);
    });
  });
});
