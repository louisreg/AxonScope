#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _RattayAberham_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"RattayAberham.mod\"");
    fprintf(stderr, "\n");
  }
  _RattayAberham_reg();
}

#if defined(__cplusplus)
}
#endif
