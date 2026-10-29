import { defineConfig, mergeConfig } from 'vitest/config'

import viteConfig from './vite.config.ts'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      restoreMocks: true,
      // jsdom + Mantine in Docker on Windows: the first test of a file also loads the page
      // code, and parallel files compete for CPU (seen: 0.9 s alone, 5.8 s in the full run).
      testTimeout: 15_000,
      coverage: {
        provider: 'v8',
        include: ['src/**/*.{ts,tsx}'],
        // Generated API types, test helpers and the entry file are not application logic.
        exclude: [
          'src/api/schema.d.ts',
          'src/test/**',
          'src/**/*.test.{ts,tsx}',
          'src/**/*.test-data.ts',
          'src/main.tsx',
        ],
        reporter: ['text-summary', 'text'],
      },
    },
  }),
)
