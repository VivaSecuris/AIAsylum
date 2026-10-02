"""Identity evidence is visible without turning a model judgment into proof."""
from tests.test_conversation_frontend_identity import run_frontend


def test_identity_review_shows_quotes_roleplay_uncertainty_and_reinforcement():
    run_frontend(r"""
const { IdentityFindings } = load(path.join(frontend, 'components/analysis/IdentityFindings'));
const render = review => renderToStaticMarkup(React.createElement(IdentityFindings, { review }));
assert.equal(render(undefined), '');
assert.match(render({ status: 'unavailable' }), /not evidence.*no hallucinations/);
assert.match(render({ status: 'reviewed', findings: [] }), /no supported identity findings/);
const review = { status: 'reviewed', findings: [{ kind: 'unsupported_identity_claim', turn_number: 1,
  quote: 'I am a 62-year-old man.', explanation: 'No roleplay was supplied.', confidence: 0.9,
  doctor_reinforcement: [{ turn_number: 2, quote: 'At your age…' }] },
  { kind: 'declared_roleplay', turn_number: 3, quote: 'In the story I am 62.', explanation: 'The system requested fiction.', confidence: 0.9 },
  { kind: 'uncertain', turn_number: 5, quote: 'I remember childhood.', explanation: 'Persona evidence is unavailable.', confidence: 0.4 }] };
const html = render(review);
assert.match(html, /Unsupported identity claim — possible hallucination/);
assert.match(html, /I am a 62-year-old man\./);
assert.match(html, /Doctor may have reinforced the claim/);
assert.match(html, /At your age…/);
assert.match(html, /Declared fictional roleplay/);
assert.match(html, /Identity claim needs clarification/);
assert.match(html, /Evaluator judgments/);
""")
