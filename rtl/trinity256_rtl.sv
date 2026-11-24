// ============================================================================
// Trinity-256 Heterogeneous AI Superchip - Complete Hierarchical RTL
// Document ID: SIL-ARCH-TRINITY256-001
// Hierarchy:
//   1. trinity256_pkg             - Global types, address map, NoC flit structs
//   2. trinity_tile_sram_2mb      - 4-banked, dual-ported 2 MB Tile SRAM
//   3. trinity_nct_tile           - Neural Compute Tile (RV64 + Coral + NVDLA)
//   4. trinity_quad_semaphores    - 64 hardware counting semaphores & IRQs
//   5. trinity_neural_quad        - 4 NCTs + 512b AXI5 Crossbar + 4 MB Quad L3
//   6. trinity_torus_router       - 1024b Folded Torus Router w/ VC2 AllReduce
//   7. trinity_compute_chiplet    - 16 Quads (64 Tiles) + 4x4 Torus + 2x HBM3E
//   8. trinity256_package_top     - 4 Chiplets + UCIe D2D Interposer + 8x HBM3E
// ============================================================================

`timescale 1ns / 1ps

package trinity256_pkg;

    // ------------------------------------------------------------------------
    // Architectural Constants
    // ------------------------------------------------------------------------
    parameter int NUM_CHIPLETS         = 4;
    parameter int QUADS_PER_CHIPLET    = 16; // 4x4 Folded Torus of Quads
    parameter int TILES_PER_QUAD       = 4;  // 2x2 Neural Quad
    parameter int TOTAL_TILES          = NUM_CHIPLETS * QUADS_PER_CHIPLET * TILES_PER_QUAD; // 256

    parameter int ADDR_WIDTH           = 64;
    parameter int TILE_DATA_WIDTH      = 128;
    parameter int QUAD_DATA_WIDTH      = 512;
    parameter int NOC_FLIT_WIDTH       = 1024;
    parameter int HBM_DATA_WIDTH       = 1024;

    // ------------------------------------------------------------------------
    // Unified 64-bit Physical Address Map Constants
    // Bit layout: [63:40]=0 | [39:38]=Chiplet | [37:34]=Quad | [33:32]=Tile | [31:0]=Offset
    // ------------------------------------------------------------------------
    localparam logic [63:0] LOCAL_CORAL_BASE  = 64'h0000_0000_0000;
    localparam logic [63:0] LOCAL_CORAL_MASK  = 64'h0000_0000_FFFF; // 64 KB (ITCM/DTCM/CSRs)

    localparam logic [63:0] LOCAL_NVDLA_BASE  = 64'h0000_0010_0000;
    localparam logic [63:0] LOCAL_NVDLA_MASK  = 64'h0000_000F_FFFF; // 1 MB CSB

    localparam logic [63:0] LOCAL_SRAM_BASE   = 64'h0000_0020_0000;
    localparam logic [63:0] LOCAL_SRAM_MASK   = 64'h0000_001F_FFFF; // 2 MB Tile SRAM

    localparam logic [63:0] QUAD_SRAM_BASE    = 64'h0000_0100_0000;
    localparam logic [63:0] QUAD_SRAM_MASK    = 64'h0000_003F_FFFF; // 4 MB Quad L3/Scratchpad

    localparam logic [63:0] QUAD_SEM_BASE     = 64'h0000_0140_0000;
    localparam logic [63:0] QUAD_SEM_MASK     = 64'h0000_0000_FFFF; // 64 KB Counting Semaphores

    localparam logic [63:0] GLOBAL_MESH_BASE  = 64'h0001_0000_0000;
    localparam logic [63:0] GLOBAL_MMIO_BASE  = 64'h0100_0000_0000;
    localparam logic [63:0] GLOBAL_HBM_BASE   = 64'h8000_0000_0000;

    // ------------------------------------------------------------------------
    // NoC Virtual Channel Definitions
    // ------------------------------------------------------------------------
    typedef enum logic [1:0] {
        VC0_COHERENCY  = 2'b00, // TileLink-C / CHI directory & control
        VC1_TENSOR_DMA = 2'b01, // Zero-copy RDMA tensor pushes
        VC2_ALLREDUCE  = 2'b10, // In-network FP16/INT32 collective summation
        VC3_MGMT_IRQ   = 2'b11  // PMU power-gating & global wake broadcasts
    } noc_vc_e;

    // ------------------------------------------------------------------------
    // 1024-bit Folded Torus NoC Flit Structure
    // ------------------------------------------------------------------------
    typedef struct packed {
        logic         valid;
        noc_vc_e      vc;
        logic [1:0]   src_chiplet;
        logic [3:0]   src_quad;
        logic [1:0]   src_tile;
        logic [1:0]   dst_chiplet;
        logic [3:0]   dst_quad;
        logic [1:0]   dst_tile;
        logic [2:0]   opcode;        // 0:RD, 1:WR, 2:RDMA_PUSH, 3:ALLREDUCE_INT32, 4:ALLREDUCE_FP16, 5:WAKE
        logic [38:0]  addr_offset;   // Target offset or HBM address
        logic         gw_hbm;        // 1: destined for the chiplet HBM gateway (dst_quad unused)
        logic [1:0]   rsvd;          // pads flit to exactly 1024 bits
        logic [959:0] payload;       // 960-bit tensor/cacheline payload
    } noc_flit_t;

    // ------------------------------------------------------------------------
    // Simplified AXI4 Master/Slave Bundle Structs (Synthesizable)
    // ------------------------------------------------------------------------
    typedef struct packed {
        logic         awvalid;
        logic [63:0]  awaddr;
        logic [7:0]   awlen;
        logic [2:0]   awsize;
        logic [1:0]   awburst;
        logic         wvalid;
        logic [127:0] wdata;
        logic [15:0]  wstrb;
        logic         wlast;
        logic         bready;
        logic         arvalid;
        logic [63:0]  araddr;
        logic [7:0]   arlen;
        logic [2:0]   arsize;
        logic [1:0]   arburst;
        logic         rready;
    } axi4_128_req_t;

    typedef struct packed {
        logic         awready;
        logic         wready;
        logic         bvalid;
        logic [1:0]   bresp;
        logic         arready;
        logic         rvalid;
        logic [127:0] rdata;
        logic [1:0]   rresp;
        logic         rlast;
    } axi4_128_resp_t;

endpackage : trinity256_pkg


// ============================================================================
// 1b. Latch-based integrated clock gate (replaces glitch-prone `clk & en`)
// ============================================================================
module trinity_clk_gate (
    input  logic clk,
    input  logic en,
    output logic gclk
);
    logic en_l;
    always_latch if (!clk) en_l = en;
    assign gclk = clk & en_l;
endmodule : trinity_clk_gate


// ============================================================================
// 2. 2 MB Dual-Ported, 4-Banked Tile-Local Shared SRAM
//    - Port A: Local Crossbar (RV64GC + Coral NPU + Quad Xbar DMA)
//    - Port B: Dedicated NVDLA SRAMIF (Zero-DRAM activation streaming)
// ============================================================================
module trinity_tile_sram_2mb (
    input  logic         clk,
    input  logic         rst_n,

    // Port A: 128-bit Local AXI4 Slave (Simplified single-beat / burst wrapper)
    input  logic         porta_en,
    input  logic         porta_we,
    input  logic [16:0]  porta_addr,   // 131,072 x 128-bit words = 2 MB
    input  logic [127:0] porta_wdata,
    input  logic [15:0]  porta_wstrb,
    output logic [127:0] porta_rdata,

    // Port B: 128-bit Dedicated NVDLA SRAMIF
    input  logic         portb_en,
    input  logic         portb_we,
    input  logic [16:0]  portb_addr,
    input  logic [127:0] portb_wdata,
    input  logic [15:0]  portb_wstrb,
    output logic [127:0] portb_rdata
);
    // 4 Banks of 512 KB (32,768 x 128-bit each) to eliminate bank conflicts
    localparam int NUM_BANKS  = 4;
    localparam int BANK_DEPTH = 32768;

    logic [1:0]  bank_sel_a, bank_sel_b;
    logic [14:0] bank_idx_a, bank_idx_b;

    assign bank_sel_a = porta_addr[1:0];
    assign bank_idx_a = porta_addr[16:2];
    assign bank_sel_b = portb_addr[1:0];
    assign bank_idx_b = portb_addr[16:2];

    logic [127:0] bank_rdata_a [NUM_BANKS];
    logic [127:0] bank_rdata_b [NUM_BANKS];

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b++) begin : gen_sram_banks
            // True Dual-Port SRAM Macro Inference
            (* ram_style = "ultra" *)
            logic [127:0] mem [0:BANK_DEPTH-1];

            // Port A  (plain `always`: true-dual-port array has two writers)
            always @(posedge clk) begin
                if (porta_en && (bank_sel_a == b[1:0])) begin
                    if (porta_we) begin
                        for (int i = 0; i < 16; i++) begin
                            if (porta_wstrb[i])
                                mem[bank_idx_a][i*8 +: 8] <= porta_wdata[i*8 +: 8];
                        end
                    end
                    bank_rdata_a[b] <= mem[bank_idx_a];
                end
            end

            // Port B (NVDLA SRAMIF)
            always @(posedge clk) begin
                if (portb_en && (bank_sel_b == b[1:0])) begin
                    if (portb_we) begin
                        for (int i = 0; i < 16; i++) begin
                            if (portb_wstrb[i])
                                mem[bank_idx_b][i*8 +: 8] <= portb_wdata[i*8 +: 8];
                        end
                    end
                    bank_rdata_b[b] <= mem[bank_idx_b];
                end
            end
        end
    endgenerate

    logic [1:0] bank_sel_a_q, bank_sel_b_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_sel_a_q <= '0;
            bank_sel_b_q <= '0;
        end else begin
            bank_sel_a_q <= bank_sel_a;
            bank_sel_b_q <= bank_sel_b;
        end
    end

    assign porta_rdata = bank_rdata_a[bank_sel_a_q];
    assign portb_rdata = bank_rdata_b[bank_sel_b_q];

endmodule : trinity_tile_sram_2mb


// ============================================================================
// 3. Neural Compute Tile (NCT) - Repeated 256 Times Across Package
//    Contains:
//      - RV64GCVB RISC-V Host Core (CVA6-class)
//      - Google Coral NPU (RV32IMF + Zve32f + Zvt 256 MAC/cycle)
//      - NVIDIA NVDLA (nv_medium_512 w/ DBB + SRAMIF)
//      - 2 MB Dual-Ported Tile SRAM
//      - Local 128-bit AXI4 Crossbar & Power-Gating Isolation
// ============================================================================
module trinity_nct_tile
    import trinity256_pkg::*;
#(
    parameter logic [1:0] CHIPLET_ID = 2'd0,
    parameter logic [3:0] QUAD_ID    = 4'd0,
    parameter logic [1:0] TILE_ID    = 2'd0
) (
    input  logic           clk,
    input  logic           rst_n,

    // Per-Tile Independent Power/Clock Gating Control
    input  logic           tile_pwr_en,        // 1: Active, 0: Power-gated
    input  logic           coral_only_mode,    // 1: P3 Sentinel Standby (RV64 & NVDLA gated)
    output logic           coral_wake_irq,     // Asserted when Coral NPU detects wake event

    // Tile-to-Quad 128-bit Uplink (Connects to 512-bit Quad Crossbar)
    output axi4_128_req_t  quad_up_req,
    input  axi4_128_resp_t quad_up_resp,

    // Quad-to-Tile 128-bit Downlink (Remote RDMA / Coherency snoops into Tile SRAM)
    input  axi4_128_req_t  quad_dn_req,
    output axi4_128_resp_t quad_dn_resp,

    // Hardware Semaphore Doorbells from Quad Controller
    input  logic           sem_coral_doorbell,
    input  logic           sem_nvdla_doorbell,
    output logic           nvdla_done_pulse,
    output logic           coral_done_pulse
);

    // Gated Clocks for Power-Proportional Scaling
    logic clk_rv64_nvdla;
    logic clk_coral;
    trinity_clk_gate u_cg_coral (.clk(clk), .en(tile_pwr_en),                       .gclk(clk_coral));
    trinity_clk_gate u_cg_main  (.clk(clk), .en(tile_pwr_en & ~coral_only_mode),    .gclk(clk_rv64_nvdla));

    // Internal AXI4 Master/Slave Buses
    axi4_128_req_t  rv64_m_req, coral_m_req, nvdla_dbb_req, nvdla_sram_req;
    axi4_128_resp_t rv64_m_resp, coral_m_resp, nvdla_dbb_resp, nvdla_sram_resp;

    // Control Space Bus (CSB) for NVDLA & Coral TCM/CSR MMIO
    logic        nvdla_csb_valid, nvdla_csb_write, nvdla_csb_ready;
    logic [15:0] nvdla_csb_addr;
    logic [31:0] nvdla_csb_wdata, nvdla_csb_rdata;

    logic        coral_mmio_valid, coral_mmio_write, coral_mmio_ready;
    logic [15:0] coral_mmio_addr;
    logic [31:0] coral_mmio_wdata, coral_mmio_rdata;

    // Tile SRAM Port A & B Signals
    logic         sram_pa_en, sram_pa_we;
    logic [16:0]  sram_pa_addr;
    logic [127:0] sram_pa_wdata, sram_pa_rdata;
    logic [15:0]  sram_pa_wstrb;

    logic         sram_pb_en, sram_pb_we;
    logic [16:0]  sram_pb_addr;
    logic [127:0] sram_pb_wdata, sram_pb_rdata;
    logic [15:0]  sram_pb_wstrb;

    // Hardwired Tile HartID: {Chiplet[1:0], Quad[3:0], Tile[1:0]} = 0..255
    logic [7:0] hart_id;
    assign hart_id = {CHIPLET_ID, QUAD_ID, TILE_ID};

    // ------------------------------------------------------------------------
    // 3.1 RISC-V RV64GCVB Host Core (CVA6 / Ariane RTL Blackbox Integration)
    // ------------------------------------------------------------------------
    rv64gcvb_core_wrapper #(
        .L1I_KB (32),
        .L1D_KB (32),
        .L2_KB  (256)
    ) u_rv64_core (
        .clk_i        (clk_rv64_nvdla),
        .rst_ni       (rst_n & tile_pwr_en & ~coral_only_mode),
        .hart_id_i    ({56'd0, hart_id}),
        .irq_ext_i    (coral_wake_irq | nvdla_done_pulse),
        .axi_req_o    (rv64_m_req),
        .axi_resp_i   (rv64_m_resp)
    );

    // ------------------------------------------------------------------------
    // 3.2 Google Coral NPU (RV32IMF + Zve32f + Zvt 256 MAC/cycle)
    // ------------------------------------------------------------------------
    coral_npu_wrapper #(
        .ITCM_KB      (8),
        .DTCM_KB      (32),
        .MACS_PER_CYC (256)
    ) u_coral_npu (
        .clk_i          (clk_coral),
        .rst_ni         (rst_n & tile_pwr_en),
        .doorbell_i     (sem_coral_doorbell),
        .wake_irq_o     (coral_wake_irq),
        .done_pulse_o   (coral_done_pulse),
        // Slave TCM/CSR MMIO programming port
        .mmio_valid_i   (coral_mmio_valid),
        .mmio_write_i   (coral_mmio_write),
        .mmio_addr_i    (coral_mmio_addr),
        .mmio_wdata_i   (coral_mmio_wdata),
        .mmio_rdata_o   (coral_mmio_rdata),
        .mmio_ready_o   (coral_mmio_ready),
        // Master AXI4 DMA port (Tile SRAM & Quad L3 access)
        .axi_m_req_o    (coral_m_req),
        .axi_m_resp_i   (coral_m_resp)
    );

    // ------------------------------------------------------------------------
    // 3.3 NVIDIA NVDLA (nv_medium_512: 512 INT8 MACs/cyc, 256 KB CBUF)
    // ------------------------------------------------------------------------
    nvdla_nv_medium_wrapper #(
        .MAC_ATOMIC_C (64),
        .MAC_ATOMIC_K (8),
        .CBUF_KB      (256)
    ) u_nvdla (
        .dla_core_clk       (clk_rv64_nvdla),
        .dla_csb_clk        (clk_rv64_nvdla),
        .global_clk_ovr_on  (1'b0),
        .tmc2slcg_disable   (1'b0),
        .dla_reset_rstn     (rst_n & tile_pwr_en & ~coral_only_mode),
        .direct_reset_      (rst_n),
        .doorbell_start_i   (sem_nvdla_doorbell),
        .dla_intr_o         (nvdla_done_pulse),
        // 32-bit Control Space Bus (CSB)
        .csb2nvdla_valid    (nvdla_csb_valid),
        .csb2nvdla_ready    (nvdla_csb_ready),
        .csb2nvdla_addr     (nvdla_csb_addr),
        .csb2nvdla_wdat     (nvdla_csb_wdata),
        .csb2nvdla_write    (nvdla_csb_write),
        .nvdla2csb_valid    (),
        .nvdla2csb_data     (nvdla_csb_rdata),
        // Data Backbone (DBB) -> Routes to HBM3E via Quad/Mesh
        .dbb_axi_req_o      (nvdla_dbb_req),
        .dbb_axi_resp_i     (nvdla_dbb_resp),
        // Secondary SRAMIF -> Binds directly to Port B of 2 MB Tile SRAM
        .sramif_axi_req_o   (nvdla_sram_req),
        .sramif_axi_resp_i  (nvdla_sram_resp)
    );

    // Bind NVDLA SRAMIF directly to Tile SRAM Port B (1-cycle deterministic access)
    assign sram_pb_en    = nvdla_sram_req.awvalid | nvdla_sram_req.arvalid;
    assign sram_pb_we    = nvdla_sram_req.awvalid;
    assign sram_pb_addr  = nvdla_sram_req.awvalid ? nvdla_sram_req.awaddr[20:4] : nvdla_sram_req.araddr[20:4];
    assign sram_pb_wdata = nvdla_sram_req.wdata;
    assign sram_pb_wstrb = nvdla_sram_req.wstrb;

    always_ff @(posedge clk_rv64_nvdla or negedge rst_n) begin
        if (!rst_n) begin
            nvdla_sram_resp <= '0;
        end else begin
            nvdla_sram_resp.awready <= 1'b1;
            nvdla_sram_resp.wready  <= 1'b1;
            nvdla_sram_resp.arready <= 1'b1;
            nvdla_sram_resp.bvalid  <= nvdla_sram_req.awvalid & nvdla_sram_req.wvalid;
            nvdla_sram_resp.bresp   <= 2'b00;
            nvdla_sram_resp.rvalid  <= nvdla_sram_req.arvalid;
            nvdla_sram_resp.rdata   <= sram_pb_rdata;
            nvdla_sram_resp.rresp   <= 2'b00;
            nvdla_sram_resp.rlast   <= nvdla_sram_req.arvalid;
        end
    end

    // ------------------------------------------------------------------------
    // 3.4 2 MB Dual-Ported Tile SRAM Instance
    // ------------------------------------------------------------------------
    trinity_tile_sram_2mb u_tile_sram (
        .clk         (clk_coral),
        .rst_n       (rst_n),
        .porta_en    (sram_pa_en),
        .porta_we    (sram_pa_we),
        .porta_addr  (sram_pa_addr),
        .porta_wdata (sram_pa_wdata),
        .porta_wstrb (sram_pa_wstrb),
        .porta_rdata (sram_pa_rdata),
        .portb_en    (sram_pb_en),
        .portb_we    (sram_pb_we),
        .portb_addr  (sram_pb_addr),
        .portb_wdata (sram_pb_wdata),
        .portb_wstrb (sram_pb_wstrb),
        .portb_rdata (sram_pb_rdata)
    );

    // ------------------------------------------------------------------------
    // 3.5 Local 128-bit AXI4 Address Decoder & Round-Robin Arbiter
    //     Masters: [0]=RV64GC, [1]=Coral NPU, [2]=NVDLA DBB, [3]=Quad Downlink
    //     Slaves : Coral TCM/CSR, NVDLA CSB, Local 2MB SRAM, Quad Uplink
    // ------------------------------------------------------------------------
    axi4_128_req_t  arb_req;
    logic [1:0]     arb_grant;

    // Priority/Round-Robin selection across local masters
    always_comb begin
        arb_grant = 2'd0;
        arb_req   = '0;
        if (quad_dn_req.awvalid || quad_dn_req.arvalid) begin
            arb_grant = 2'd3;
            arb_req   = quad_dn_req;
        end else if (coral_m_req.awvalid || coral_m_req.arvalid) begin
            arb_grant = 2'd1;
            arb_req   = coral_m_req;
        end else if (nvdla_dbb_req.awvalid || nvdla_dbb_req.arvalid) begin
            arb_grant = 2'd2;
            arb_req   = nvdla_dbb_req;
        end else if (rv64_m_req.awvalid || rv64_m_req.arvalid) begin
            arb_grant = 2'd0;
            arb_req   = rv64_m_req;
        end
    end

    logic [63:0] active_addr;
    logic        is_write, is_read;
    assign is_write    = arb_req.awvalid;
    assign is_read     = arb_req.arvalid;
    assign active_addr = is_write ? arb_req.awaddr : arb_req.araddr;

    // Decode Local Tile Alias vs. Off-Tile (Quad/NoC/HBM)
    logic sel_coral_mmio, sel_nvdla_csb, sel_local_sram, sel_quad_up;
    assign sel_coral_mmio = (active_addr[63:16] == 48'h0000_0000_0000);
    assign sel_nvdla_csb  = (active_addr[63:20] == 44'h0000_0000_001);
    assign sel_local_sram = (active_addr[63:21] == 43'h0000_0000_001); // 0x20_0000..0x3F_FFFF
    assign sel_quad_up    = ~(sel_coral_mmio | sel_nvdla_csb | sel_local_sram);

    // Route to Coral NPU MMIO
    assign coral_mmio_valid = (is_write | is_read) & sel_coral_mmio;
    assign coral_mmio_write = is_write;
    assign coral_mmio_addr  = active_addr[15:0];
    assign coral_mmio_wdata = arb_req.wdata[31:0];

    // Route to NVDLA CSB
    assign nvdla_csb_valid  = (is_write | is_read) & sel_nvdla_csb;
    assign nvdla_csb_write  = is_write;
    assign nvdla_csb_addr   = active_addr[15:0];
    assign nvdla_csb_wdata  = arb_req.wdata[31:0];

    // Route to Tile SRAM Port A
    assign sram_pa_en       = (is_write | is_read) & sel_local_sram;
    assign sram_pa_we       = is_write;
    assign sram_pa_addr     = active_addr[20:4];
    assign sram_pa_wdata    = arb_req.wdata;
    assign sram_pa_wstrb    = arb_req.wstrb;

    // Route to Quad Uplink (Tag source Tile ID into unused upper metadata or sideband)
    always_comb begin
        quad_up_req         = arb_req;
        quad_up_req.awvalid = arb_req.awvalid & sel_quad_up;
        quad_up_req.arvalid = arb_req.arvalid & sel_quad_up;
    end

    // Return response mux to granted master
    axi4_128_resp_t mux_resp;
    always_comb begin
        mux_resp = '0;
        if (sel_quad_up) begin
            mux_resp = quad_up_resp;
        end else if (sel_local_sram) begin
            mux_resp.awready = 1'b1;
            mux_resp.wready  = 1'b1;
            mux_resp.arready = 1'b1;
            mux_resp.bvalid  = is_write;
            // read data phase is returned one cycle later (see rd_resp below)
        end else if (sel_nvdla_csb) begin
            mux_resp.awready = nvdla_csb_ready;
            mux_resp.wready  = nvdla_csb_ready;
            mux_resp.arready = nvdla_csb_ready;
            mux_resp.bvalid  = is_write & nvdla_csb_ready;
            mux_resp.rvalid  = is_read  & nvdla_csb_ready;
            mux_resp.rdata   = {96'd0, nvdla_csb_rdata};
            mux_resp.rlast   = is_read;
        end else if (sel_coral_mmio) begin
            mux_resp.awready = coral_mmio_ready;
            mux_resp.wready  = coral_mmio_ready;
            mux_resp.arready = coral_mmio_ready;
            mux_resp.bvalid  = is_write & coral_mmio_ready;
            mux_resp.rvalid  = is_read  & coral_mmio_ready;
            mux_resp.rdata   = {96'd0, coral_mmio_rdata};
            mux_resp.rlast   = is_read;
        end
    end

    // Registered read-data phase for the 1-cycle-latency SRAM Port A
    logic           sram_rd_q;
    logic [1:0]     sram_rd_grant_q;
    axi4_128_resp_t rd_resp;
    always_ff @(posedge clk_coral or negedge rst_n) begin
        if (!rst_n) begin
            sram_rd_q       <= 1'b0;
            sram_rd_grant_q <= 2'd0;
        end else begin
            sram_rd_q       <= sram_pa_en & ~sram_pa_we;
            sram_rd_grant_q <= arb_grant;
        end
    end
    always_comb begin
        rd_resp        = '0;
        rd_resp.rvalid = sram_rd_q;
        rd_resp.rlast  = sram_rd_q;
        rd_resp.rdata  = sram_pa_rdata;
    end

    assign rv64_m_resp    = ((arb_grant == 2'd0) ? mux_resp : '0) | ((sram_rd_q && sram_rd_grant_q == 2'd0) ? rd_resp : '0);
    assign coral_m_resp   = ((arb_grant == 2'd1) ? mux_resp : '0) | ((sram_rd_q && sram_rd_grant_q == 2'd1) ? rd_resp : '0);
    assign nvdla_dbb_resp = ((arb_grant == 2'd2) ? mux_resp : '0) | ((sram_rd_q && sram_rd_grant_q == 2'd2) ? rd_resp : '0);
    assign quad_dn_resp   = ((arb_grant == 2'd3) ? mux_resp : '0) | ((sram_rd_q && sram_rd_grant_q == 2'd3) ? rd_resp : '0);

endmodule : trinity_nct_tile


// ============================================================================
// 4. Hardware Counting Semaphores & Zero-Overhead Doorbell Controller
//    - 64 atomic counters per Neural Quad
//    - Automatically triggers Coral NPU / NVDLA start when count reaches 0
// ============================================================================
module trinity_quad_semaphores #(
    parameter int NUM_SEMS = 64
) (
    input  logic        clk,
    input  logic        rst_n,

    // MMIO Write/Read from Quad Crossbar
    input  logic        req_valid,
    input  logic        req_we,
    input  logic [5:0]  sem_idx,
    input  logic [31:0] req_wdata,   // [15:0]=Init/Dec value, [19:16]=Target Tile, [21:20]=Engine (1:Coral, 2:NVDLA)
    output logic [31:0] resp_rdata,

    // Hardware Auto-Decrement Pulses from Tile Engines
    input  logic [3:0]  nvdla_done,
    input  logic [3:0]  coral_done,

    // Hardware Doorbell Outputs to the 4 Tiles
    output logic [3:0]  coral_doorbell,
    output logic [3:0]  nvdla_doorbell
);
    logic [15:0] count      [NUM_SEMS];
    logic [1:0]  dst_tile   [NUM_SEMS];
    logic [1:0]  dst_engine [NUM_SEMS];
    logic        armed      [NUM_SEMS];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            coral_doorbell <= '0;
            nvdla_doorbell <= '0;
            for (int i = 0; i < NUM_SEMS; i++) begin
                count[i]      <= 16'd0;
                dst_tile[i]   <= 2'd0;
                dst_engine[i] <= 2'd0;
                armed[i]      <= 1'b0;
            end
        end else begin
            coral_doorbell <= '0;
            nvdla_doorbell <= '0;

            // Software Programming / Manual Decrement
            if (req_valid && req_we) begin
                if (req_wdata[31]) begin
                    // Bit [31]=1: Initialize & arm counting semaphore
                    count[sem_idx]      <= req_wdata[15:0];
                    dst_tile[sem_idx]   <= req_wdata[17:16];
                    dst_engine[sem_idx] <= req_wdata[21:20];
                    armed[sem_idx]      <= 1'b1;
                end else if (armed[sem_idx] && count[sem_idx] > 16'd0) begin
                    // Bit [31]=0: Decrement counter by 1
                    count[sem_idx] <= count[sem_idx] - 16'd1;
                    if (count[sem_idx] == 16'd1) begin
                        armed[sem_idx] <= 1'b0;
                        if (dst_engine[sem_idx] == 2'd1)
                            coral_doorbell[dst_tile[sem_idx]] <= 1'b1;
                        else if (dst_engine[sem_idx] == 2'd2)
                            nvdla_doorbell[dst_tile[sem_idx]] <= 1'b1;
                    end
                end
            end

            // Hardware auto-decrement on NVDLA / Coral completion (Semaphores 0..7 mapped)
            for (int t = 0; t < 4; t++) begin
                if (nvdla_done[t] && armed[t] && count[t] > 16'd0) begin
                    count[t] <= count[t] - 16'd1;
                    if (count[t] == 16'd1) begin
                        armed[t] <= 1'b0;
                        if (dst_engine[t] == 2'd1) coral_doorbell[dst_tile[t]] <= 1'b1;
                        if (dst_engine[t] == 2'd2) nvdla_doorbell[dst_tile[t]] <= 1'b1;
                    end
                end
                if (coral_done[t] && armed[t+4] && count[t+4] > 16'd0) begin
                    count[t+4] <= count[t+4] - 16'd1;
                    if (count[t+4] == 16'd1) begin
                        armed[t+4] <= 1'b0;
                        if (dst_engine[t+4] == 2'd1) coral_doorbell[dst_tile[t+4]] <= 1'b1;
                        if (dst_engine[t+4] == 2'd2) nvdla_doorbell[dst_tile[t+4]] <= 1'b1;
                    end
                end
            end
        end
    end

    assign resp_rdata = {10'd0, armed[sem_idx], 1'b0, dst_engine[sem_idx], dst_tile[sem_idx], count[sem_idx]};

endmodule : trinity_quad_semaphores


// ============================================================================
// 5. Neural Quad (4 Tiles + 512-bit AXI5 Crossbar + 4 MB Shared Quad L3)
//    - Repeated 64 times across the 256-core package
// ============================================================================
module trinity_neural_quad
    import trinity256_pkg::*;
#(
    parameter logic [1:0] CHIPLET_ID = 2'd0,
    parameter logic [3:0] QUAD_ID    = 4'd0
) (
    input  logic        clk,
    input  logic        rst_n,

    // Power-Gating Controls per Tile in Quad
    input  logic [3:0]  tile_pwr_en,
    input  logic [3:0]  coral_only_mode,
    output logic [3:0]  quad_wake_irq,

    // Connection to 4x4 Chiplet Folded Torus Router
    output noc_flit_t   noc_tx_flit,
    input  logic        noc_tx_ready,
    input  noc_flit_t   noc_rx_flit,
    output logic        noc_rx_ready
);

    axi4_128_req_t  tile_up_req  [4];
    axi4_128_resp_t tile_up_resp [4];
    axi4_128_req_t  tile_dn_req  [4];
    axi4_128_resp_t tile_dn_resp [4];

    logic [3:0] coral_doorbell, nvdla_doorbell;
    logic [3:0] nvdla_done, coral_done;

    // ------------------------------------------------------------------------
    // 5.1 Instantiate 4 Neural Compute Tiles (2x2)
    // ------------------------------------------------------------------------
    genvar t;
    generate
        for (t = 0; t < TILES_PER_QUAD; t++) begin : gen_tiles
            trinity_nct_tile #(
                .CHIPLET_ID (CHIPLET_ID),
                .QUAD_ID    (QUAD_ID),
                .TILE_ID    (t[1:0])
            ) u_nct (
                .clk                (clk),
                .rst_n              (rst_n),
                .tile_pwr_en        (tile_pwr_en[t]),
                .coral_only_mode    (coral_only_mode[t]),
                .coral_wake_irq     (quad_wake_irq[t]),
                .quad_up_req        (tile_up_req[t]),
                .quad_up_resp       (tile_up_resp[t]),
                .quad_dn_req        (tile_dn_req[t]),
                .quad_dn_resp       (tile_dn_resp[t]),
                .sem_coral_doorbell (coral_doorbell[t]),
                .sem_nvdla_doorbell (nvdla_doorbell[t]),
                .nvdla_done_pulse   (nvdla_done[t]),
                .coral_done_pulse   (coral_done[t])
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 5.2 4 MB Shared Quad SRAM / L3 Slice (65,536 x 512-bit wide)
    //     Single-cycle intra-quad access for NVDLA <-> Coral tensor fusion
    // ------------------------------------------------------------------------
    (* ram_style = "ultra" *)
    logic [511:0] quad_l3_mem [0:65535];
    logic         quad_sram_en, quad_sram_we;
    logic [15:0]  quad_sram_addr;
    logic [511:0] quad_sram_wdata, quad_sram_rdata;
    logic [63:0]  quad_sram_wstrb;

    always_ff @(posedge clk) begin
        if (quad_sram_en) begin
            if (quad_sram_we) begin
                for (int i = 0; i < 64; i++)
                    if (quad_sram_wstrb[i])
                        quad_l3_mem[quad_sram_addr][i*8 +: 8] <= quad_sram_wdata[i*8 +: 8];
            end
            quad_sram_rdata <= quad_l3_mem[quad_sram_addr];
        end
    end

    // ------------------------------------------------------------------------
    // 5.3 Quad Hardware Counting Semaphores
    // ------------------------------------------------------------------------
    logic        sem_req_valid, sem_req_we;
    logic [5:0]  sem_idx;
    logic [31:0] sem_wdata, sem_rdata;

    trinity_quad_semaphores #(
        .NUM_SEMS (64)
    ) u_quad_sems (
        .clk            (clk),
        .rst_n          (rst_n),
        .req_valid      (sem_req_valid),
        .req_we         (sem_req_we),
        .sem_idx        (sem_idx),
        .req_wdata      (sem_wdata),
        .resp_rdata     (sem_rdata),
        .nvdla_done     (nvdla_done),
        .coral_done     (coral_done),
        .coral_doorbell (coral_doorbell),
        .nvdla_doorbell (nvdla_doorbell)
    );

    // ------------------------------------------------------------------------
    // 5.4 512-bit Quad Crossbar & NoC Packetizer / Depacketizer
    //     One tile is granted per cycle; only the granted tile is acknowledged.
    //     Local (SRAM / semaphore) reads return data one cycle after the grant.
    // ------------------------------------------------------------------------
    logic [1:0]      rr_tile_ptr;
    logic [1:0]      sel_idx;
    axi4_128_req_t   sel_up_req;
    logic            any_up_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rr_tile_ptr <= 2'd0;
        else        rr_tile_ptr <= rr_tile_ptr + 2'd1;
    end

    always_comb begin
        any_up_valid = 1'b0;
        sel_up_req   = '0;
        sel_idx      = 2'd0;
        for (int i = 0; i < 4; i++) begin
            logic [1:0] idx;
            idx = rr_tile_ptr + i[1:0];
            if (!any_up_valid && (tile_up_req[idx].awvalid || tile_up_req[idx].arvalid)) begin
                any_up_valid = 1'b1;
                sel_up_req   = tile_up_req[idx];
                sel_idx      = idx;
            end
        end
    end

    logic [63:0] up_addr;
    logic        up_is_wr, up_is_rd;
    logic [1:0]  up_lane;
    assign up_is_wr = sel_up_req.awvalid;
    assign up_is_rd = sel_up_req.arvalid;
    assign up_addr  = up_is_wr ? sel_up_req.awaddr : sel_up_req.araddr;
    assign up_lane  = up_addr[5:4];

    // Decode Local Quad SRAM (0x0100_0000..0x013F_FFFF), Semaphores (0x0140_0000), or Global NoC
    logic is_quad_sram, is_quad_sem, is_global_noc;
    assign is_quad_sram  = (up_addr[63:22] == 42'h0000_0000_04);
    assign is_quad_sem   = (up_addr[63:16] == 48'h0000_0140);
    assign is_global_noc = ~(is_quad_sram | is_quad_sem);

    // Quad L3: 128-bit beat lands in lane up_addr[5:4] of the 512-bit line, byte-enabled
    assign quad_sram_en    = any_up_valid & is_quad_sram;
    assign quad_sram_we    = up_is_wr;
    assign quad_sram_addr  = up_addr[21:6];
    assign quad_sram_wdata = {4{sel_up_req.wdata}};
    assign quad_sram_wstrb = 64'(sel_up_req.wstrb) << {up_lane, 4'd0};

    // Semaphore MMIO hookup (32-bit word selected by addr[3:2])
    assign sem_req_valid   = any_up_valid & is_quad_sem;
    assign sem_req_we      = up_is_wr;
    assign sem_idx         = up_addr[7:2];
    assign sem_wdata       = sel_up_req.wdata[{up_addr[3:2], 5'd0} +: 32];

    // Packetize Global Mesh / HBM3E requests onto a 1024-bit NoC flit.
    // Mesh addresses carry only the 32-bit local offset (coordinates are in the header);
    // HBM addresses keep all 39 bits. HBM is flagged by gw_hbm, not by dst_quad.
    always_comb begin
        noc_tx_flit             = '0;
        noc_tx_flit.valid       = any_up_valid & is_global_noc;
        noc_tx_flit.vc          = up_addr[63] ? VC1_TENSOR_DMA : VC0_COHERENCY;
        noc_tx_flit.src_chiplet = CHIPLET_ID;
        noc_tx_flit.src_quad    = QUAD_ID;
        noc_tx_flit.src_tile    = sel_idx;
        noc_tx_flit.dst_chiplet = up_addr[63] ? CHIPLET_ID : up_addr[39:38];
        noc_tx_flit.dst_quad    = up_addr[63] ? 4'd0       : up_addr[37:34];
        noc_tx_flit.dst_tile    = up_addr[33:32];
        noc_tx_flit.gw_hbm      = up_addr[63];
        noc_tx_flit.opcode      = up_is_wr ? 3'd2 : 3'd0; // RDMA_PUSH or RD
        noc_tx_flit.addr_offset = up_addr[63] ? up_addr[38:0] : {7'd0, up_addr[31:0]};
        noc_tx_flit.payload     = {832'd0, sel_up_req.wdata};
    end

    // Depacketize Incoming NoC Flits destined for this Quad's Tiles.
    // Offset is the 32-bit local alias; the upper coordinate bits must NOT reach the
    // tile decoder or the access would be mis-routed back up the uplink.
    assign noc_rx_ready = 1'b1;
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            tile_dn_req[i] = '0;
            if (noc_rx_flit.valid && (noc_rx_flit.dst_tile == i[1:0])) begin
                tile_dn_req[i].awvalid = (noc_rx_flit.opcode == 3'd1 || noc_rx_flit.opcode == 3'd2 ||
                                          noc_rx_flit.opcode == 3'd3 || noc_rx_flit.opcode == 3'd4);
                tile_dn_req[i].awaddr  = {32'd0, noc_rx_flit.addr_offset[31:0]};
                tile_dn_req[i].wvalid  = tile_dn_req[i].awvalid;
                tile_dn_req[i].wdata   = noc_rx_flit.payload[127:0];
                tile_dn_req[i].wstrb   = 16'hFFFF;
                tile_dn_req[i].wlast   = 1'b1;
                tile_dn_req[i].arvalid = (noc_rx_flit.opcode == 3'd0);
                tile_dn_req[i].araddr  = {32'd0, noc_rx_flit.addr_offset[31:0]};
                tile_dn_req[i].rready  = 1'b1;
            end
        end
    end

    // Registered read-data phase for local reads
    logic        gnt_rd_q, gnt_sem_q;
    logic [1:0]  gnt_idx_q, gnt_lane_q;
    logic [31:0] sem_rdata_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gnt_rd_q    <= 1'b0;
            gnt_sem_q   <= 1'b0;
            gnt_idx_q   <= 2'd0;
            gnt_lane_q  <= 2'd0;
            sem_rdata_q <= 32'd0;
        end else begin
            gnt_rd_q    <= any_up_valid & up_is_rd & ~is_global_noc;
            gnt_sem_q   <= is_quad_sem;
            gnt_idx_q   <= sel_idx;
            gnt_lane_q  <= up_lane;
            sem_rdata_q <= sem_rdata;
        end
    end

    // Acknowledge ONLY the granted tile (previously every requesting tile was acked)
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            tile_up_resp[i] = '0;
            if (any_up_valid && sel_idx == i[1:0]) begin
                tile_up_resp[i].awready = is_global_noc ? noc_tx_ready : 1'b1;
                tile_up_resp[i].wready  = is_global_noc ? noc_tx_ready : 1'b1;
                tile_up_resp[i].arready = is_global_noc ? noc_tx_ready : 1'b1;
                tile_up_resp[i].bvalid  = up_is_wr & (is_global_noc ? noc_tx_ready : 1'b1);
            end
            if (gnt_rd_q && gnt_idx_q == i[1:0]) begin
                tile_up_resp[i].rvalid = 1'b1;
                tile_up_resp[i].rlast  = 1'b1;
                tile_up_resp[i].rdata  = gnt_sem_q ? {96'd0, sem_rdata_q}
                                                   : quad_sram_rdata[{gnt_lane_q, 7'd0} +: 128];
            end
        end
    end

endmodule : trinity_neural_quad


// ============================================================================
// 6. 1024-bit 4x4 Folded Torus Router w/ In-Network VC2 AllReduce ALU
//    - Dimension-order (X then Y) shortest-path routing on a 4x4 torus
//    - Gateway traffic (HBM / other chiplet) is steered to column x==3 and
//      leaves on east_out, where the chiplet taps it (see trinity_compute_chiplet)
//    - `edge_in` injects flits arriving from the UCIe bridges (x==3 routers)
//    - VC2 INT32 AllReduce: absorbs ALLREDUCE_FANIN contributions, then emits
//      their lane-wise sum and clears (accumulator used to grow forever)
//    - Transit flits have priority; local injection is back-pressured through
//      local_in_ready instead of being silently dropped.
//    KNOWN LIMITATION: no input FIFOs, so two transit flits wanting the router
//    in the same cycle still collide (lower-priority one is lost). Real hardware
//    needs buffered virtual channels with credit flow control.
// ============================================================================
module trinity_torus_router
    import trinity256_pkg::*;
#(
    parameter logic [1:0] CHIPLET_ID       = 2'd0,
    parameter logic [1:0] X_COORD          = 2'd0,
    parameter logic [1:0] Y_COORD          = 2'd0,
    parameter int         ALLREDUCE_FANIN  = 4
) (
    input  logic      clk,
    input  logic      rst_n,

    // Local Quad Injection / Ejection
    input  noc_flit_t local_in,
    output logic      local_in_ready,
    output noc_flit_t local_out,
    input  logic      local_out_ready,

    // Torus Cardinal Ports (North, South, East, West) + UCIe edge injection
    input  noc_flit_t north_in, south_in, east_in, west_in, edge_in,
    output noc_flit_t north_out, south_out, east_out, west_out
);
    logic transit_valid;
    assign transit_valid  = west_in.valid | east_in.valid | north_in.valid |
                            south_in.valid | edge_in.valid;
    assign local_in_ready = ~transit_valid;

    noc_flit_t selected_flit;
    always_comb begin
        if      (west_in.valid)  selected_flit = west_in;
        else if (east_in.valid)  selected_flit = east_in;
        else if (north_in.valid) selected_flit = north_in;
        else if (south_in.valid) selected_flit = south_in;
        else if (edge_in.valid)  selected_flit = edge_in;
        else if (local_in.valid) selected_flit = local_in;
        else                     selected_flit = '0;
    end

    // ---- In-network INT32 AllReduce (VC2, opcode 3): 30 x 32-bit lanes ----
    logic [959:0] allreduce_accum;
    logic [7:0]   allreduce_cnt;
    logic         is_ar, ar_last;
    assign is_ar   = selected_flit.valid && selected_flit.vc == VC2_ALLREDUCE &&
                     selected_flit.opcode == 3'd3;
    assign ar_last = is_ar && (allreduce_cnt == 8'(unsigned'(ALLREDUCE_FANIN - 1)));

    logic [959:0] reduced_payload;
    always_comb begin
        reduced_payload = selected_flit.payload;
        if (is_ar) begin
            for (int lane = 0; lane < 30; lane++) begin
                reduced_payload[lane*32 +: 32] =
                    selected_flit.payload[lane*32 +: 32] + allreduce_accum[lane*32 +: 32];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            allreduce_accum <= '0;
            allreduce_cnt   <= '0;
        end else if (is_ar) begin
            if (ar_last) begin
                allreduce_accum <= '0;
                allreduce_cnt   <= '0;
            end else begin
                allreduce_accum <= reduced_payload;
                allreduce_cnt   <= allreduce_cnt + 8'd1;
            end
        end
    end

    // Partial contributions are absorbed; only the final sum continues
    noc_flit_t routed_flit;
    always_comb begin
        routed_flit         = selected_flit;
        routed_flit.payload = reduced_payload;
        if (is_ar && !ar_last) routed_flit.valid = 1'b0;
    end

    // ---- Dimension-order shortest-path torus routing ----
    logic [1:0] dst_x, dst_y;
    logic       is_gw;
    assign dst_x = routed_flit.dst_quad[1:0];
    assign dst_y = routed_flit.dst_quad[3:2];
    assign is_gw = routed_flit.gw_hbm || (routed_flit.dst_chiplet != CHIPLET_ID);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_out <= '0;
            north_out <= '0;
            south_out <= '0;
            east_out  <= '0;
            west_out  <= '0;
        end else begin
            local_out <= '0;
            north_out <= '0;
            south_out <= '0;
            east_out  <= '0;
            west_out  <= '0;

            if (routed_flit.valid) begin
                if (is_gw) begin
                    // Head for the gateway column x==3 by the shortest wrap direction
                    if (X_COORD == 2'd3)                          east_out <= routed_flit;
                    else if (((2'd3 - X_COORD) & 2'b11) <= 2'd2)  east_out <= routed_flit;
                    else                                          west_out <= routed_flit;
                end else if ((dst_x == X_COORD) && (dst_y == Y_COORD)) begin
                    if (local_out_ready) local_out <= routed_flit;
                end else if (dst_x != X_COORD) begin
                    if (((dst_x - X_COORD) & 2'b11) <= 2'd2) east_out <= routed_flit;
                    else                                     west_out <= routed_flit;
                end else begin
                    if (((dst_y - Y_COORD) & 2'b11) <= 2'd2) south_out <= routed_flit;
                    else                                     north_out <= routed_flit;
                end
            end
        end
    end

endmodule : trinity_torus_router


// ============================================================================
// 7. Compute Chiplet (16 Neural Quads = 64 Tiles + 4x4 Folded Torus + 2x HBM3E)
//    - Repeated 4 times on the 2.5D CoWoS / UCIe silicon interposer
// ============================================================================
module trinity_compute_chiplet
    import trinity256_pkg::*;
#(
    parameter logic [1:0] CHIPLET_ID = 2'd0
) (
    input  logic         clk,
    input  logic         rst_n,

    // Chiplet-Wide Per-Tile Power-Gating Vector (64 Tiles)
    input  logic [63:0]  chiplet_tile_pwr_en,
    input  logic [63:0]  chiplet_coral_only,
    output logic         chiplet_wake_irq,

    // UCIe Die-to-Die Inter-Chiplet Links (Horizontal & Vertical Neighbor Chiplets)
    output noc_flit_t    ucie_h_tx,
    input  noc_flit_t    ucie_h_rx,
    output noc_flit_t    ucie_v_tx,
    input  noc_flit_t    ucie_v_rx,

    // 2x HBM3E Stack Interfaces per Chiplet (1024-bit wide each = 2.4 TB/s per chiplet)
    output logic         hbm0_req_valid,
    output logic         hbm0_req_we,
    output logic [33:0]  hbm0_req_addr,
    output logic [1023:0] hbm0_req_wdata,
    input  logic [1023:0] hbm0_resp_rdata,

    output logic         hbm1_req_valid,
    output logic         hbm1_req_we,
    output logic [33:0]  hbm1_req_addr,
    output logic [1023:0] hbm1_req_wdata,
    input  logic [1023:0] hbm1_resp_rdata
);

    // Gateway-bound flits are consumed at the x==3 tap; they must not wrap to x==0
    function automatic noc_flit_t strip_gw(noc_flit_t f);
        if (f.valid && (f.gw_hbm || f.dst_chiplet != CHIPLET_ID)) return '0;
        return f;
    endfunction

    noc_flit_t quad_tx [4][4];
    noc_flit_t quad_rx [4][4];
    logic      quad_tx_rdy [4][4];
    logic      quad_rx_rdy [4][4];

    noc_flit_t n_in [4][4], s_in [4][4], e_in [4][4], w_in [4][4];
    noc_flit_t n_out[4][4], s_out[4][4], e_out[4][4], w_out[4][4];

    logic [15:0] quad_irqs;
    assign chiplet_wake_irq = |quad_irqs;

    // ------------------------------------------------------------------------
    // 7.1 Instantiate 4x4 Grid of Neural Quads + Torus Routers (64 Tiles)
    // ------------------------------------------------------------------------
    genvar x, y;
    generate
        for (y = 0; y < 4; y++) begin : gen_torus_y
            for (x = 0; x < 4; x++) begin : gen_torus_x
                localparam int Q_IDX = y * 4 + x;
                logic [3:0] w_irq_vec;
                assign quad_irqs[Q_IDX] = |w_irq_vec;

                // UCIe ingress joins the torus at the gateway column
                noc_flit_t edge_in_w;
                if (x == 3 && y == 2)      assign edge_in_w = ucie_h_rx;
                else if (x == 3 && y == 3) assign edge_in_w = ucie_v_rx;
                else                       assign edge_in_w = '0;

                trinity_neural_quad #(
                    .CHIPLET_ID (CHIPLET_ID),
                    .QUAD_ID    (Q_IDX[3:0])
                ) u_quad (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .tile_pwr_en     (chiplet_tile_pwr_en[Q_IDX*4 +: 4]),
                    .coral_only_mode (chiplet_coral_only[Q_IDX*4 +: 4]),
                    .quad_wake_irq   (w_irq_vec),
                    .noc_tx_flit     (quad_tx[y][x]),
                    .noc_tx_ready    (quad_tx_rdy[y][x]),
                    .noc_rx_flit     (quad_rx[y][x]),
                    .noc_rx_ready    (quad_rx_rdy[y][x])
                );

                trinity_torus_router #(
                    .CHIPLET_ID (CHIPLET_ID),
                    .X_COORD    (x[1:0]),
                    .Y_COORD    (y[1:0])
                ) u_router (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .local_in        (quad_tx[y][x]),
                    .local_in_ready  (quad_tx_rdy[y][x]),
                    .local_out       (quad_rx[y][x]),
                    .local_out_ready (quad_rx_rdy[y][x]),
                    .north_in        (n_in[y][x]),
                    .south_in        (s_in[y][x]),
                    .east_in         (e_in[y][x]),
                    .west_in         (w_in[y][x]),
                    .edge_in         (edge_in_w),
                    .north_out       (n_out[y][x]),
                    .south_out       (s_out[y][x]),
                    .east_out        (e_out[y][x]),
                    .west_out        (w_out[y][x])
                );

                // Wrap-around 4x4 Folded Torus Wiring
                assign n_in[y][x] = s_out[(y + 3) % 4][x];
                assign s_in[y][x] = n_out[(y + 1) % 4][x];
                assign w_in[y][x] = (x == 0) ? strip_gw(e_out[y][(x + 3) % 4]) : e_out[y][(x + 3) % 4];
                assign e_in[y][x] = w_out[y][(x + 1) % 4];
            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // 7.2 Chiplet Edge Gateways: 2x HBM3E Controllers & UCIe Die-to-Die Bridges
    //     Every row's x==3 router is a tap. Rows 0-1 -> HBM stack 0, rows 2-3 ->
    //     HBM stack 1 (previously rows 2/3 had no HBM path). Off-chiplet flits go to
    //     UCIe: horizontal first (chiplet id bit 0), then vertical (bit 1).
    //     Lowest row index wins if several taps fire in one cycle.
    // ------------------------------------------------------------------------
    noc_flit_t hbm0_flit, hbm1_flit, ucie_flit;
    always_comb begin
        hbm0_flit = '0;
        hbm1_flit = '0;
        ucie_flit = '0;
        for (int yy = 3; yy >= 0; yy--) begin
            if (e_out[yy][3].valid) begin
                if (e_out[yy][3].gw_hbm) begin
                    if (yy < 2) hbm0_flit = e_out[yy][3];
                    else        hbm1_flit = e_out[yy][3];
                end else if (e_out[yy][3].dst_chiplet != CHIPLET_ID) begin
                    ucie_flit = e_out[yy][3];
                end
            end
        end
    end

    assign hbm0_req_valid = hbm0_flit.valid;
    assign hbm0_req_we    = (hbm0_flit.opcode == 3'd1 || hbm0_flit.opcode == 3'd2);
    assign hbm0_req_addr  = hbm0_flit.addr_offset[33:0];
    assign hbm0_req_wdata = {64'd0, hbm0_flit.payload};

    assign hbm1_req_valid = hbm1_flit.valid;
    assign hbm1_req_we    = (hbm1_flit.opcode == 3'd1 || hbm1_flit.opcode == 3'd2);
    assign hbm1_req_addr  = hbm1_flit.addr_offset[33:0];
    assign hbm1_req_wdata = {64'd0, hbm1_flit.payload};

    // Diagonal traffic leaves horizontally; the neighbour forwards it vertically
    assign ucie_h_tx = (ucie_flit.valid && ucie_flit.dst_chiplet[0] != CHIPLET_ID[0]) ? ucie_flit : '0;
    assign ucie_v_tx = (ucie_flit.valid && ucie_flit.dst_chiplet[0] == CHIPLET_ID[0]) ? ucie_flit : '0;

    // HBM read-data return path is not modelled in this wrapper
    logic unused_hbm;
    assign unused_hbm = (^hbm0_resp_rdata) ^ (^hbm1_resp_rdata);

endmodule : trinity_compute_chiplet


// ============================================================================
// 8. Top-Level Trinity-256 Package (4 Chiplets + UCIe Interposer + 8x HBM3E)
//    - 256 RISC-V RV64GCVB Cores + 256 Coral NPUs + 256 NVDLAs + 768 MB SRAM
// ============================================================================
module trinity256_package_top
    import trinity256_pkg::*;
(
    input  logic         sys_clk_1200mhz,
    input  logic         sys_rst_n,

    // Global Power Management Unit (PMU) Control & Wake Interface
    input  logic [1:0]   pkg_power_state,   // 0:P0 Full(300W), 1:P1 SRAM, 2:P2 1-Chiplet, 3:P3 Sentinel(<15W)
    output logic         pkg_global_wake_irq,

    // PCIe Gen5 x16 Host Control Interface (Simplified MMIO / DMA Doorbell)
    input  logic         pcie_cfg_valid,
    input  logic [31:0]  pcie_cfg_addr,
    input  logic [31:0]  pcie_cfg_wdata,

    // 8x HBM3E Physical Memory Stack Interfaces (8 x 1024-bit = 9.6 TB/s)
    output logic [7:0]   hbm_req_valid,
    output logic [7:0]   hbm_req_we,
    output logic [33:0]  hbm_req_addr  [8],
    output logic [1023:0] hbm_req_wdata [8],
    input  logic [1023:0] hbm_resp_rdata[8]
);

    // ------------------------------------------------------------------------
    // 8.1 Global PMU Power-Gating & Staggered Wake Sequencer (di/dt Safe)
    // ------------------------------------------------------------------------
    logic [255:0] global_tile_pwr_en;
    logic [255:0] global_coral_only;
    logic [3:0]   chiplet_wake_irqs;

    assign pkg_global_wake_irq = |chiplet_wake_irqs;

    always_ff @(posedge sys_clk_1200mhz or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            // Boot into P3 Sentinel Standby: Only Tile[0,0,0] Coral NPU active (<15W)
            global_tile_pwr_en <= 256'h0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001;
            global_coral_only  <= 256'hFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;
        end else begin
            case (pkg_power_state)
                2'd0: begin // P0: Full 256-Tile Saturation (472 TOPS)
                    global_tile_pwr_en <= {256{1'b1}};
                    global_coral_only  <= {256{1'b0}};
                end
                2'd1: begin // P1: SRAM-Resident Mode
                    global_tile_pwr_en <= {256{1'b1}};
                    global_coral_only  <= {256{1'b0}};
                end
                2'd2: begin // P2: Single-Chiplet (64 Tiles Active, 192 Tiles Power-Gated)
                    global_tile_pwr_en <= {{192{1'b0}}, {64{1'b1}}};
                    global_coral_only  <= {256{1'b0}};
                end
                default: begin // P3: Sentinel Standby (<15W: 1 Coral NPU active, 255 Tiles gated)
                    global_tile_pwr_en <= {{255{1'b0}}, 1'b1};
                    global_coral_only  <= {256{1'b1}};
                end
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // 8.2 2x2 Chiplet Array & UCIe Die-to-Die Interposer Mesh
    //     Chiplet 0 (Top-Left)  <--UCIe--> Chiplet 1 (Top-Right)
    //         ^                                ^
    //        UCIe                             UCIe
    //         v                                v
    //     Chiplet 2 (Bot-Left)  <--UCIe--> Chiplet 3 (Bot-Right)
    // ------------------------------------------------------------------------
    noc_flit_t ucie_h_tx [4], ucie_h_rx [4];
    noc_flit_t ucie_v_tx [4], ucie_v_rx [4];

    // Cross-connect UCIe Horizontal & Vertical Die-to-Die Interposer Channels
    assign ucie_h_rx[0] = ucie_h_tx[1];
    assign ucie_h_rx[1] = ucie_h_tx[0];
    assign ucie_h_rx[2] = ucie_h_tx[3];
    assign ucie_h_rx[3] = ucie_h_tx[2];

    assign ucie_v_rx[0] = ucie_v_tx[2];
    assign ucie_v_rx[2] = ucie_v_tx[0];
    assign ucie_v_rx[1] = ucie_v_tx[3];
    assign ucie_v_rx[3] = ucie_v_tx[1];

    genvar c;
    generate
        for (c = 0; c < NUM_CHIPLETS; c++) begin : gen_chiplets
            trinity_compute_chiplet #(
                .CHIPLET_ID (c[1:0])
            ) u_chiplet (
                .clk                 (sys_clk_1200mhz),
                .rst_n               (sys_rst_n),
                .chiplet_tile_pwr_en (global_tile_pwr_en[c*64 +: 64]),
                .chiplet_coral_only  (global_coral_only[c*64 +: 64]),
                .chiplet_wake_irq    (chiplet_wake_irqs[c]),
                .ucie_h_tx           (ucie_h_tx[c]),
                .ucie_h_rx           (ucie_h_rx[c]),
                .ucie_v_tx           (ucie_v_tx[c]),
                .ucie_v_rx           (ucie_v_rx[c]),
                .hbm0_req_valid      (hbm_req_valid[c*2]),
                .hbm0_req_we         (hbm_req_we[c*2]),
                .hbm0_req_addr       (hbm_req_addr[c*2]),
                .hbm0_req_wdata      (hbm_req_wdata[c*2]),
                .hbm0_resp_rdata     (hbm_resp_rdata[c*2]),
                .hbm1_req_valid      (hbm_req_valid[c*2 + 1]),
                .hbm1_req_we         (hbm_req_we[c*2 + 1]),
                .hbm1_req_addr       (hbm_req_addr[c*2 + 1]),
                .hbm1_req_wdata      (hbm_req_wdata[c*2 + 1]),
                .hbm1_resp_rdata     (hbm_resp_rdata[c*2 + 1])
            );
        end
    endgenerate

    logic unused_pcie;
    assign unused_pcie = pcie_cfg_valid ^ (^pcie_cfg_addr) ^ (^pcie_cfg_wdata);

endmodule : trinity256_package_top


// ============================================================================
// 9. Open-Source IP Integration Wrappers (Synthesis / Simulation Stubs)
//    Replace module bodies with generated Chisel/Verilog RTL from:
//      - OpenHW CVA6 / UC Berkeley Rocket (RV64GCVB)
//      - Google Coral NPU (Chisel -> Verilog)
//      - NVIDIA NVDLA (nv_medium_512 Verilog)
// ============================================================================
module rv64gcvb_core_wrapper
    import trinity256_pkg::*;
#(
    parameter int L1I_KB = 32,
    parameter int L1D_KB = 32,
    parameter int L2_KB  = 256
) (
    input  logic           clk_i,
    input  logic           rst_ni,
    input  logic [63:0]    hart_id_i,
    input  logic           irq_ext_i,
    output axi4_128_req_t  axi_req_o,
    input  axi4_128_resp_t axi_resp_i
);
    // Behavioral heartbeat fetch stub for clean lint/elaboration
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            axi_req_o <= '0;
        end else begin
            axi_req_o.arvalid <= 1'b0;
            axi_req_o.awvalid <= 1'b0;
            axi_req_o.rready  <= 1'b1;
            axi_req_o.bready  <= 1'b1;
        end
    end
    logic unused_rv;
    assign unused_rv = (^hart_id_i) ^ irq_ext_i ^ axi_resp_i.rvalid;
endmodule

module coral_npu_wrapper
    import trinity256_pkg::*;
#(
    parameter int ITCM_KB      = 8,
    parameter int DTCM_KB      = 32,
    parameter int MACS_PER_CYC = 256
) (
    input  logic           clk_i,
    input  logic           rst_ni,
    input  logic           doorbell_i,
    output logic           wake_irq_o,
    output logic           done_pulse_o,
    input  logic           mmio_valid_i,
    input  logic           mmio_write_i,
    input  logic [15:0]    mmio_addr_i,
    input  logic [31:0]    mmio_wdata_i,
    output logic [31:0]    mmio_rdata_o,
    output logic           mmio_ready_o,
    output axi4_128_req_t  axi_m_req_o,
    input  axi4_128_resp_t axi_m_resp_i
);
    logic [31:0] ctrl_reg;
    assign mmio_ready_o = 1'b1;
    assign mmio_rdata_o = ctrl_reg;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ctrl_reg     <= 32'd0;
            wake_irq_o   <= 1'b0;
            done_pulse_o <= 1'b0;
            axi_m_req_o  <= '0;
        end else begin
            done_pulse_o <= doorbell_i;
            if (mmio_valid_i && mmio_write_i && (mmio_addr_i == 16'h0000)) begin
                ctrl_reg   <= mmio_wdata_i;
                wake_irq_o <= mmio_wdata_i[0];
            end
        end
    end
    logic unused_coral;
    assign unused_coral = axi_m_resp_i.rvalid;
endmodule

module nvdla_nv_medium_wrapper
    import trinity256_pkg::*;
#(
    parameter int MAC_ATOMIC_C = 64,
    parameter int MAC_ATOMIC_K = 8,
    parameter int CBUF_KB      = 256
) (
    input  logic           dla_core_clk,
    input  logic           dla_csb_clk,
    input  logic           global_clk_ovr_on,
    input  logic           tmc2slcg_disable,
    input  logic           dla_reset_rstn,
    input  logic           direct_reset_,
    input  logic           doorbell_start_i,
    output logic           dla_intr_o,
    input  logic           csb2nvdla_valid,
    output logic           csb2nvdla_ready,
    input  logic [15:0]    csb2nvdla_addr,
    input  logic [31:0]    csb2nvdla_wdat,
    input  logic           csb2nvdla_write,
    output logic           nvdla2csb_valid,
    output logic [31:0]    nvdla2csb_data,
    output axi4_128_req_t  dbb_axi_req_o,
    input  axi4_128_resp_t dbb_axi_resp_i,
    output axi4_128_req_t  sramif_axi_req_o,
    input  axi4_128_resp_t sramif_axi_resp_i
);
    logic [31:0] csb_reg;
    assign csb2nvdla_ready = 1'b1;

    always_ff @(posedge dla_core_clk or negedge dla_reset_rstn) begin
        if (!dla_reset_rstn) begin
            csb_reg          <= 32'd0;
            dla_intr_o       <= 1'b0;
            nvdla2csb_valid  <= 1'b0;
            nvdla2csb_data   <= 32'd0;
            dbb_axi_req_o    <= '0;
            sramif_axi_req_o <= '0;
        end else begin
            dla_intr_o      <= doorbell_start_i;
            nvdla2csb_valid <= csb2nvdla_valid & ~csb2nvdla_write;
            nvdla2csb_data  <= csb_reg;
            if (csb2nvdla_valid && csb2nvdla_write) begin
                csb_reg <= csb2nvdla_wdat;
            end
        end
    end
    logic unused_nvdla;
    assign unused_nvdla = dla_csb_clk ^ global_clk_ovr_on ^ tmc2slcg_disable ^
                          direct_reset_ ^ (^csb2nvdla_addr) ^
                          dbb_axi_resp_i.rvalid ^ sramif_axi_resp_i.rvalid;
endmodule