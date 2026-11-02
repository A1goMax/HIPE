# HIPE Current Method Evidence

MS04 is the active manuscript. The requested scope is Tables 1-5; earlier biological simulation plans are outside this implementation campaign.

| Claim/reference | Required evidence | Current code state |
| --- | --- | --- |
| Table 1 retrieval | Exact generator, sequence/batch settings and 20k-step accuracy/loss | Sources identified; adapter migration pending |
| Table 2 restricted-budget LM | Four methods, matched protocol, 100M tokens, original batches and explicit sigma selection | Shared code/suite implemented; no new paper metrics |
| Table 3 300M scaling | 1B-token L512/L2048 runs for RoPE, fixed/learnable HIPE and XPos | Shared variants supported; exact campaign settings pending |
| Table 4 SST-2 | Pretrained checkpoint identities, LoRA/classifier protocol and paired subsets | Separate adapter integration pending |
| Table 5 attention | Six 60M/L1024/300M-token conditions; HIPE sigma 500 | Shared model/trainer with shallow/deep local layer lists |

Registry numbers are reported values. Historical commit fields and filenames remain search leads until the executed source/config/data association is established.

The current manuscript describes unscaled RoPE as the sharp-field reference. The old planning critique of a direct sigma-zero identity should not silently change this implementation. The fixed HIPE formula is preserved; unscaled RoPE is selected explicitly.
