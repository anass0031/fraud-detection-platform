import type { CaseDetail } from '../features/case/api'
import type { CaseRow } from '../features/queue/api'

/** A queue row like the API returns (demo case 122 by default). */
export function caseRow(overrides: Partial<CaseRow> = {}): CaseRow {
  return {
    id: 122,
    transaction_id: 3479772,
    transaction_time: '2018-04-28T19:52:25',
    amount: 994,
    product_code: 'W',
    card_network: 'mastercard',
    fraud_score: 0.4838,
    expected_loss: 480.8878,
    risk_level: 'high',
    reason: 'model_score',
    priority_tier: 1,
    status: 'new',
    assignee: null,
    final_decision: null,
    created_at: '2026-10-28T13:29:26Z',
    updated_at: '2026-10-28T13:29:26Z',
    closed_at: null,
    ...overrides,
  }
}

/** A case detail like the API returns (demo case 94: $400, critical, real factors). */
export function caseDetail(overrides: Partial<CaseDetail> = {}): CaseDetail {
  const row = caseRow({
    id: 94,
    transaction_id: 3477470,
    transaction_time: '2018-04-28T08:37:56',
    amount: 400,
    product_code: 'R',
    card_network: 'visa',
    fraud_score: 0.9015,
    expected_loss: 360.6,
    risk_level: 'critical',
  })
  return {
    ...row,
    transaction: {
      transaction_id: 3477470,
      transaction_time: '2018-04-28T08:37:56',
      amount: 400,
      product_code: 'R',
      card_network: 'visa',
      card_type: 'credit',
      payer_email_domain: 'gmail.com',
      device_type: 'mobile',
    },
    prediction: {
      model_version: 'lgbm-2026.10.27-1',
      fraud_score: 0.9015,
      expected_loss: 360.6,
      risk_level: 'critical',
      is_alert: true,
      is_unusual: false,
      anomaly_score: 0.452,
      base_value: -4.774,
      raw_log_odds: 2.936,
      other_contribution: -0.0765,
      top_factors: FACTORS_94,
      scored_at: '2026-10-28T09:15:00Z',
    },
    card_history: [
      {
        transaction_id: 3401200,
        transaction_time: '2018-04-20T10:00:00',
        amount: 35.5,
        product_code: 'R',
        had_case: false,
      },
      {
        transaction_id: 3300100,
        transaction_time: '2018-03-02T22:15:00',
        amount: 400,
        product_code: 'R',
        had_case: true,
      },
    ],
    decisions: [],
    ...overrides,
  }
}

type Factor = CaseDetail['prediction']['top_factors'][number]
const f = (
  key: string,
  label: string,
  contribution: number,
  strength: string,
  text: string,
  masked = false,
): Factor => ({
  key,
  masked,
  label,
  value: null,
  contribution,
  direction: contribution > 0 ? 'raises risk' : 'lowers risk',
  strength,
  text,
})

export const FACTORS_94: Factor[] = [
  f(
    'masked_counts',
    'Counting signals (masked)',
    3.068,
    'strongly',
    'Provider counting signals (anonymised): strongly raises risk',
    true,
  ),
  f('amt_log', 'Amount', 1.4598, 'strongly', 'Amount $400: strongly raises risk'),
  f(
    'masked_timing',
    'Account timing (masked)',
    1.2303,
    'strongly',
    'Provider account timing signals (anonymised): strongly raises risk',
    true,
  ),
  f(
    'r_email',
    'Recipient e-mail',
    1.1976,
    'strongly',
    'Recipient e-mail domain is gmail.com: strongly raises risk',
  ),
  f(
    'device_type',
    'Device type',
    0.279,
    'moderately',
    'Device type is mobile: moderately raises risk',
  ),
  f(
    'email_same_domain',
    'E-mail match',
    0.2266,
    'moderately',
    'Payer and recipient use the same e-mail domain: moderately raises risk',
  ),
  f('ProductCD', 'Product', -0.219, 'moderately', 'Product code is R: moderately lowers risk'),
  f('browser_family', 'Browser', 0.1914, 'slightly', 'Browser is safari: slightly raises risk'),
  f(
    'p_email',
    'Payer e-mail',
    0.1841,
    'slightly',
    'Payer e-mail domain is gmail.com: slightly raises risk',
  ),
  f(
    'amt_to_uid_mean',
    'Amount vs usual',
    0.1687,
    'slightly',
    'No earlier amounts to compare with: slightly raises risk',
  ),
]
