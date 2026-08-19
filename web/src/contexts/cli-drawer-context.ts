import { createContext, useContext } from "react";

export interface CliDrawerContextValue {
  open: boolean;
  setOpen: (open: boolean) => void;
}

export const CliDrawerContext = createContext<CliDrawerContextValue>({
  open: false,
  setOpen: () => {},
});

export function useCliDrawer() {
  return useContext(CliDrawerContext);
}
