MODULE Etna

    PERS tooldata toolEtna:=[TRUE,[[215.7,-22.4,473.8],[0.500000,0.000000,-0.866025,0.000000]],[20,[70,30,123.5],[0,0,1,0],1,0,1]];
    PERS wobjdata wobjEtna:=[FALSE,TRUE,"",[[1655.0,-87.0,932.0],[1.000000,0.000000,0.000000,0.000000]],[[0,0,0],[1,0,0,0]]];

    VAR triggdata laserON;
    VAR triggdata laserOFF;

    CONST speeddata vl:=[8,500,5000,1000];

    
    CONST robtarget T0:=[[20.999964,26.839387,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T1:=[[20.999964,33.160617,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T2:=[[22.499970,35.946625,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T3:=[[22.499970,24.053379,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T4:=[[23.999976,22.537581,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T5:=[[23.999976,37.462423,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T6:=[[25.499982,38.475740,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T7:=[[25.499982,21.524264,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T8:=[[26.999988,20.852739,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T9:=[[26.999988,39.147265,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T10:=[[28.499994,39.476883,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T11:=[[28.499994,20.523119,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T12:=[[30.000000,20.416625,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T13:=[[30.000000,39.583377,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T14:=[[31.500006,39.476883,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T15:=[[31.500006,20.523119,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T16:=[[33.000012,20.852739,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T17:=[[33.000012,39.147265,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T18:=[[34.500018,38.475740,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T19:=[[34.500018,21.524264,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T20:=[[36.000024,22.537581,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T21:=[[36.000024,37.462423,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T22:=[[37.500030,35.946625,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T23:=[[37.500030,24.053379,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T24:=[[39.000036,26.839387,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T25:=[[39.000036,33.160616,0.249975],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T26:=[[38.250033,27.215177,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T27:=[[38.250033,32.784826,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T28:=[[36.750027,35.538754,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T29:=[[36.750027,24.461250,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T30:=[[35.250021,23.019748,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T31:=[[35.250021,36.980256,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T32:=[[33.750015,37.899287,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T33:=[[33.750015,22.100716,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T34:=[[32.250009,21.535383,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T35:=[[32.250009,38.464621,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T36:=[[30.750003,38.750027,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T37:=[[30.750003,21.249975,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T38:=[[29.249997,21.249975,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T39:=[[29.249997,38.750027,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T40:=[[27.749991,38.464620,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T41:=[[27.749991,21.535383,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T42:=[[26.249985,22.100716,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T43:=[[26.249985,37.899287,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T44:=[[24.749979,36.980256,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T45:=[[24.749979,23.019748,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T46:=[[23.249973,24.461250,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T47:=[[23.249973,35.538754,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T48:=[[21.749967,32.784827,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T49:=[[21.749967,27.215178,0.749985],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T50:=[[22.499970,27.590969,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T51:=[[22.499970,32.409037,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T52:=[[23.999976,35.130883,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T53:=[[23.999976,24.869122,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T54:=[[25.499982,23.501915,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T55:=[[25.499982,36.498089,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T56:=[[26.999988,37.322835,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T57:=[[26.999988,22.677169,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T58:=[[28.499994,22.227054,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T59:=[[28.499994,37.772948,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T60:=[[30.000000,37.916677,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T61:=[[30.000000,22.083326,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T62:=[[31.500006,22.227055,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T63:=[[31.500006,37.772949,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T64:=[[33.000012,37.322835,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T65:=[[33.000012,22.677169,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T66:=[[34.500018,23.501915,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T67:=[[34.500018,36.498089,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T68:=[[36.000024,35.130882,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T69:=[[36.000024,24.869121,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T70:=[[37.500030,27.590967,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T71:=[[37.500030,32.409035,1.249995],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T72:=[[36.000024,26.282941,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T73:=[[36.000024,33.717065,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T74:=[[34.500018,35.468012,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T75:=[[34.500018,24.531991,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T76:=[[33.000012,23.589384,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T77:=[[33.000012,36.410620,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T78:=[[31.500006,36.920982,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T79:=[[31.500006,23.079023,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T80:=[[30.000000,22.916676,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T81:=[[30.000000,37.083327,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T82:=[[28.499994,36.920981,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T83:=[[28.499994,23.079021,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T84:=[[26.999988,23.589384,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T85:=[[26.999988,36.410620,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T86:=[[25.499982,35.468012,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T87:=[[25.499982,24.531991,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T88:=[[23.999976,26.282939,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T89:=[[23.999976,33.717063,1.750005],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T90:=[[24.749979,26.643441,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T91:=[[24.749979,33.356562,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T92:=[[26.249985,34.985846,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T93:=[[26.249985,25.014158,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T94:=[[27.749991,24.165837,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T95:=[[27.749991,35.834167,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T96:=[[29.249997,36.229309,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T97:=[[29.249997,23.770694,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T98:=[[30.750003,23.770694,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T99:=[[30.750003,36.229309,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T100:=[[32.250009,35.834167,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T101:=[[32.250009,24.165837,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T102:=[[33.750015,25.014158,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T103:=[[33.750015,34.985846,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T104:=[[35.250021,33.356563,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T105:=[[35.250021,26.643442,2.250015],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T106:=[[34.500018,27.003943,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T107:=[[34.500018,32.996061,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T108:=[[33.000012,34.503679,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T109:=[[33.000012,25.496325,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T110:=[[31.500006,24.782956,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T111:=[[31.500006,35.217048,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T112:=[[30.000000,35.416627,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T113:=[[30.000000,24.583377,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T114:=[[28.499994,24.782956,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T115:=[[28.499994,35.217048,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T116:=[[26.999988,34.503679,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T117:=[[26.999988,25.496325,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T118:=[[25.499982,27.003943,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget T119:=[[25.499982,32.996061,2.750025],[1.000000,0.000000,0.000000,0.000000],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

PROC cladding()
    !Reset doLDLStartST;

    TriggL T0,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T1,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T2,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T3,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T4,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T5,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T6,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T7,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T8,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T9,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T10,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T11,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T12,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T13,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T14,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T15,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T16,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T17,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T18,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T19,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T20,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T21,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T22,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T23,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T24,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T25,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T26,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T27,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T28,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T29,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T30,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T31,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T32,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T33,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T34,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T35,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T36,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T37,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T38,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T39,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T40,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T41,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T42,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T43,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T44,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T45,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T46,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T47,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T48,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T49,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T50,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T51,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T52,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T53,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T54,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T55,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T56,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T57,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T58,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T59,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T60,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T61,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T62,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T63,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T64,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T65,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T66,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T67,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T68,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T69,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T70,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T71,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T72,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T73,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T74,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T75,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T76,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T77,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T78,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T79,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T80,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T81,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T82,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T83,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T84,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T85,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T86,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T87,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T88,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T89,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T90,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T91,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T92,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T93,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T94,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T95,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T96,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T97,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T98,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T99,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T100,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T101,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T102,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T103,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T104,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T105,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T106,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T107,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T108,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T109,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T110,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T111,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T112,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T113,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T114,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T115,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T116,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T117,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;
    TriggL T118,v50,laserON,z0,toolEtna\WObj:=wobjEtna;
    TriggL T119,vl,laserOFF,z0,toolEtna\WObj:=wobjEtna;

!Reset doLDLStartST;
ENDPROC

PROC mainEtna()
    Set Do_RF_MainOn;
    Set Do_RF_StandByOn;
    WaitDI DI_RF_LaserBeamReady,1;
    WaitDI DI_RF_GeneralFault,0;

    SetGO GO_Program_Rf, 0;
    WaitTime 1;
    !SetGO GoLDL_Pwr3, 1200;

    TriggIO laserON, 0\DOp:=Do_RF_ExterGate, 1;
    TriggIO laserOFF, 0\DOp:=Do_RF_ExterGate, 0;

    Set DoWeldGas;
    !MedicoatL2 "OFF", 5, 20, 7.5;
    MedicoatL1 "OFF", 5, 20, 15;

    ConfL \Off;

    MoveL [[0.0,0.0,100.0],[1.0,0.0,0.0,0.0],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v80,z0,toolEtna\WObj:=wobjEtna;

    cladding;

    MoveL [[0.0,0.0,100.0],[1.0,0.0,0.0,0.0],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v80,z0,toolEtna\WObj:=wobjEtna;

    Reset doMdtPL2On;
    Reset doMdtPL1On;
    Reset DoWeldGas;

    Reset Do_RF_StandByOn;
    !Reset Do_RF_MainOn;

ENDPROC

ENDMODULE