declare module "smiles-drawer" {
  type ParseSuccess = (tree: unknown) => void;
  type ParseError = (error: unknown) => void;

  interface DrawerOptions {
    width?: number;
    height?: number;
    bondThickness?: number;
    bondLength?: number;
    shortBondLength?: number;
    compactDrawing?: boolean;
    explicitHydrogens?: boolean;
    terminalCarbons?: boolean;
    padding?: number;
  }

  class Drawer {
    constructor(options?: DrawerOptions);
    draw(
      tree: unknown,
      target: HTMLCanvasElement | string,
      theme?: "light" | "dark",
      infoOnly?: boolean,
    ): void;
  }

  const SmilesDrawer: {
    Drawer: typeof Drawer;
    parse(smiles: string, success: ParseSuccess, error?: ParseError): void;
  };

  export default SmilesDrawer;
}
